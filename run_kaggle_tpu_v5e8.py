# numpy compatibility patch for librosa/numba on newer numpy
import numpy as _np
if not hasattr(_np, "row_stack"):
    _np.row_stack = _np.vstack
if not hasattr(_np, "trapz"):
    _np.trapz = _np.trapezoid
if not hasattr(_np, "in1d"):
    _np.in1d = lambda ar1, ar2, assume_unique=False, invert=False, kind=None: _np.isin(
        ar1, ar2, assume_unique=assume_unique, invert=invert, kind=kind
    )
if not hasattr(_np, "complex"):
    _np.complex = complex
if not hasattr(_np, "float"):
    _np.float = float
if not hasattr(_np, "int"):
    _np.int = int
if not hasattr(_np, "bool"):
    _np.bool = bool
if not hasattr(_np, "object"):
    _np.object = object
if not hasattr(_np, "str"):
    _np.str = str

import numpy as _np
if not hasattr(_np, "row_stack"):
    _np.row_stack = _np.vstack
if not hasattr(_np, "trapz"):
    _np.trapz = _np.trapezoid
if not hasattr(_np, "complex"):
    _np.complex = complex
if not hasattr(_np, "float"):
    _np.float = float
if not hasattr(_np, "int"):
    _np.int = int

"""
LongCat-Video-Avatar 1.5 on Kaggle TPU v5e-8 (single process, PyTorch/XLA GSPMD).

Placement (v5e-8: 8 chips x 16 GB HBM = 128 GB, big host RAM):
  * TPU  : bf16 DiT (~31.7 GB) sharded Megatron-style over an 8-way 'model'
           mesh (~4 GB weights/chip). The DMD distillation LoRA is merged into
           the weights on the host, so the TPU graph has no LoRA branches.
  * TPU  : Wan VAE (fp32, replicated ~0.5 GB/chip) with graph-stable frame
           loops (--vae_on_tpu, default). The stock loops grew the output via
           torch.cat each frame -> per-frame recompiles; patched loops collect
           chunks in a list and cat once. --no-vae_on_tpu falls back to CPU.
  * HOST : vocal separator (onnx), Whisper-large-v3, umT5-xxl text encoder.

Graph discipline (what makes XLA viable here):
  * torch_xla.sync() after every denoise step via the pipeline's
    _after_denoise_step hook -> one compiled graph per step shape. The whole
    run needs only ~3 distinct graphs (segment-1 denoise step, avc
    cache-build forward, avc denoise step).
  * Persistent compilation cache (xr.initialize_cache) -> later segments and
    later Kaggle sessions skip compilation entirely.
  * Distill mode: 8 steps, guidance 1.0 -> CFG off, batch 1, static shapes.
  * Self-attention uses a chunked, GSPMD-annotated implementation
    (longcat_video/xla_utils.py); F.sdpa on XLA would materialize the full
    37k x 37k score matrix.

Profiling:
  --profile starts the XLA profiler server and captures a trace during
  segment-1 denoising into --profile_dir (view with tensorboard +
  tensorboard-plugin-profile), and prints torch_xla's metrics report
  (compile count / transfer / execute times).

Example:
  python run_kaggle_tpu_v5e8.py \
      --checkpoint_dir /kaggle/temp/weights/LongCat-Video-Avatar-1.5 \
      --base_dir /kaggle/temp/weights/LongCat-Video \
      --input_json assets/avatar/single_example_1.json \
      --output_dir /kaggle/working/outputs_avatar \
      --stage_1 ai2v --num_segments 4 --profile
"""

import os
import shutil
import gc
import json
import math
import time
import time as _time
import random
import argparse
import threading
from pathlib import Path

import numpy as np
import PIL.Image
import torch

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str, default='assets/avatar/single_example_1.json')
    parser.add_argument('--output_dir', type=str, default='./outputs_avatar_tpu')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to LongCat-Video-Avatar-1.5 weights')
    parser.add_argument('--base_dir', type=str, default=None,
                        help='Path to LongCat-Video base weights (tokenizer/text_encoder/vae). '
                             'Defaults to <checkpoint_dir>/../LongCat-Video')
    parser.add_argument('--resolution', type=str, default='480p', choices=['480p', '720p'])
    parser.add_argument('--stage_1', type=str, default='ai2v', choices=['ai2v', 'at2v'])
    parser.add_argument('--num_segments', type=str, default='1',
                        help='1 segment = 93 frames (~3.7 s @25fps); each extra segment adds 80 frames (~3.2 s). '
                             'Use "auto" to automatically calculate from audio duration.')
    parser.add_argument('--num_inference_steps', type=int, default=8,
                        help='8 is the native DMD-distilled step count for v1.5')
    parser.add_argument('--ref_img_index', type=int, default=10)
    parser.add_argument('--mask_frame_range', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_fps', type=int, default=25)
    parser.add_argument('--xla_cache_dir', type=str, default='/kaggle/working/xla_cache',
                        help='persistent XLA compilation cache (survives restarts)')
    parser.add_argument('--graph_debug', action='store_true',
                        help='debug WHY graphs miss the persistent cache: sets PT_XLA_DEBUG_LEVEL=2 '
                             'so torch_xla prints a Compilation Analysis block (graph hash, input '
                             'count, and the python frames that produced the graph) for EVERY '
                             'compile. Run twice with this on and diff the step-0 analysis blocks '
                             'to find which op bakes a run-dependent constant. Verbose; debug only.')
    parser.add_argument('--profile', action='store_true',
                        help='capture an XLA profiler trace during segment 1 + print metrics report')
    parser.add_argument('--profile_dir', type=str, default='/kaggle/working/xla_profile')
    parser.add_argument('--profile_duration_ms', type=int, default=120000)
    parser.add_argument('--sync_every_n_steps', type=int, default=1,
                        help='graph-cut cadence during denoising: 1 = safest for HBM; '
                             '2/4 reduce sync barriers if memory allows')
    parser.add_argument('--runtime_scalar_fix', action=argparse.BooleanOptionalAction, default=True,
                        help='feed per-step scalars (timestep, scheduler sigma/dt) to the device as '
                             'runtime inputs so all denoise steps share ONE compiled graph, instead '
                             'of baking per-step constants into the HLO and recompiling every step. '
                             'Disable with --no-runtime_scalar_fix to get the old behavior.')
    parser.add_argument('--offload_kv_cache', action=argparse.BooleanOptionalAction, default=True,
                        help='keep the avc KV cache on HOST instead of HBM. The RESOURCE_EXHAUSTED '
                             'failure in segment 2 came from the per-chip HBM budget: the avc graph '
                             'needs ~8.5G of runtime buffers per chip, and an on-device KV cache '
                             'left too little free. 16 GB/chip, not 128 GB, is the real limit.')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=False,
                        help='resume from <output_dir>/resume_state.pt: skip segments that already '
                             'finished in a previous (crashed/interrupted) run. State (latents, '
                             'frames, audio offset, RNG) is checkpointed after every segment.')
    parser.add_argument('--vae_on_tpu', action=argparse.BooleanOptionalAction, default=True,
                        help='run the Wan VAE on the TPU mesh instead of host CPU. Default is bf16 '
                             'full-frame decode to avoid fp32 tiled-decode compile explosion. Use '
                             '--no-vae_on_tpu to fall back to host CPU.')
    parser.add_argument('--vae_dtype', type=str, default='bf16', choices=['bf16', 'fp32'],
                        help='dtype for VAE on TPU. bf16 is the fast default and should fit full-frame '
                             '480p decode; fp32 preserves original VAE precision but may need tiled decode.')
    parser.add_argument('--vae_spatial_shard', action=argparse.BooleanOptionalAction, default=True,
                        help='GSPMD-shard the full-frame VAE decode along the latent WIDTH axis over the '
                             '8-way mesh. Without this the VAE is replicated: all 8 chips compute the '
                             'identical decode (~23s/frame = single-chip throughput, 7 chips idle). The '
                             'latent width is edge-padded up to a multiple of the mesh size (92->96 for '
                             '480p) so the shards are even, and the decoded frames are cropped back. '
                             'First run after enabling recompiles the decode graphs once (new shapes + '
                             'shardings); after that they come from the persistent cache as usual.')
    parser.add_argument('--vae_tiled_decode', action=argparse.BooleanOptionalAction, default=False,
                        help='use spatial tiled VAE decode on TPU. This avoids fp32 full-frame OOM but '
                             'can compile many tile graphs; kept as fallback, not default.')
    parser.add_argument('--watchdog_interval', type=int, default=30,
                        help='seconds between background cache-size/RAM log lines, printed into this '
                             'same cell (Jupyter/Kaggle cannot run a second cell in parallel while this '
                             'one is busy, so this is how you watch cache growth + RAM live without a '
                             'separate terminal). 0 disables.')
    parser.add_argument('--warmup', action=argparse.BooleanOptionalAction, default=True,
                        help='run one throwaway 2-step ai2v/avc + VAE pass before stage 4 to force all '
                             'compilation up front. 2 steps is enough (step 0 and the steady-state graph '
                             'that steps 1-7 all share); this absorbs every first-time compile so the '
                             'real, timed segments below are pure execution time instead of a mix of '
                             'compile + compute. Disable with --no-warmup if the cache is already hot.')
    return parser.parse_args()


def torch_gc():
    gc.collect()


def generate_random_uid():
    return str(int(time.time()))[-6:] + str(random.randint(100000, 999999))


def extract_vocal_from_speech(source_path, target_path, vocal_separator, tmp_dir):
    outputs = vocal_separator.separate(source_path)
    if len(outputs) <= 0:
        print("Audio separation failed. Using raw audio.")
        return None
    default_vocal_path = (Path(tmp_dir) / "vocals" / outputs[0]).resolve().as_posix()
    shutil.move(default_vocal_path, target_path)
    return target_path


def main():
    args = _parse_args()

    # ---- XLA runtime: persistent cache + SPMD, before any computation ----
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    if args.graph_debug:
        # must be set BEFORE torch_xla import: level 2 prints a Compilation
        # Analysis block (graph hash + python frames) for every compile, which
        # is the ground truth for "why did this graph miss the cache".
        os.environ["PT_XLA_DEBUG_LEVEL"] = "2"
    import torch_xla
    import torch_xla.runtime as xr
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.debug.metrics as met

    os.makedirs(args.xla_cache_dir, exist_ok=True)
    _cache_active = False
    try:
        import torch_xla._XLAC as _xlac
        _cache_active = _xlac._xla_computation_cache_is_initialized()
    except Exception:
        pass
    if _cache_active:
        # Already initialized (second run in same kernel session). The cache
        # is still active from the first xr.initialize_cache call — we just
        # can't re-initialize it. Env vars are already set.
        _cache_path = os.environ.get('XLA_PERSISTENT_CACHE_PATH', '?')
        print(f"[xla-cache] persistent cache already active from prior run "
              f"(path={_cache_path}); new compiles will be stored/loaded", flush=True)
    else:
        try:
            xr.initialize_cache(args.xla_cache_dir, readonly=False)
            _cache_active = True
            print(f"[xla-cache] persistent cache initialized: {args.xla_cache_dir}", flush=True)
        except Exception as e:
            print(f"[xla] compilation cache unavailable: {e}")
    # ---- cache observability: never guess hit/miss from step timings again --
    _cache_files = [p for p in Path(args.xla_cache_dir).rglob("*") if p.is_file()]
    print(f"[xla-cache] dir={args.xla_cache_dir} files={len(_cache_files)} "
          f"size={sum(p.stat().st_size for p in _cache_files) / 1024**3:.2f}GB", flush=True)
    print(f"[versions] torch={torch.__version__} "
          f"torch_xla={getattr(torch_xla, '__version__', 'unknown')}", flush=True)

    def log_xla_cache_stats(tag):
        try:
            hit = met.counter_value('PersistentCacheHit') or 0
            miss = met.counter_value('PersistentCacheMiss') or 0
            compiles = met.metric_data('CompileTime')
            n_compiles = compiles[0] if compiles else 0
            print(f"[xla-cache:{tag}] persistent hit={hit} miss={miss} "
                  f"compiles_this_run={n_compiles}", flush=True)
        except Exception as e:
            print(f"[xla-cache:{tag}] stats unavailable: {e}", flush=True)
    # ---- per-sync-point cache probe: attributes every persistent-cache miss
    # to the exact step / VAE frame that produced it, instead of the coarse
    # 30s watchdog line. A line is printed ONLY when the counters moved.
    _cache_probe = {"hit": 0, "miss": 0}

    def cache_delta(tag):
        try:
            hit = met.counter_value('PersistentCacheHit') or 0
            miss = met.counter_value('PersistentCacheMiss') or 0
        except Exception:
            return
        dh = hit - _cache_probe["hit"]
        dm = miss - _cache_probe["miss"]
        _cache_probe["hit"], _cache_probe["miss"] = hit, miss
        if dh or dm:
            print(f"      [cache:{tag}] +hit={dh} +miss={dm} "
                  f"(total hit={hit} miss={miss})", flush=True)

    xr.use_spmd()
    num_devices = xr.global_runtime_device_count()
    assert num_devices >= 2, f"expected a TPU slice, found {num_devices} device(s)"
    mesh = xs.Mesh(np.arange(num_devices), (num_devices,), ("model",))
    xla_dev = torch_xla.device() if hasattr(torch_xla, "device") else xm.xla_device()
    print(f"[xla] SPMD on, 1D mesh 'model' over {num_devices} devices, device={xla_dev}")

    from longcat_video import xla_utils
    xla_utils.set_global_mesh(mesh)

    def log_mem(tag):
        parts = []
        try:
            import psutil
            parts.append(f"cpu {psutil.Process().memory_info().rss / 1024**3:.1f}GB rss")
        except Exception:
            pass
        try:
            info = xm.get_memory_info(xla_dev)
            used = info.get("bytes_used", None)
            if used is None and "bytes_available" in info:
                used = info["bytes_limit"] - info["bytes_available"]
            if used is not None:
                parts.append(f"hbm/core {used / 1024**3:.1f}/{info['bytes_limit'] / 1024**3:.1f}GB")
        except Exception:
            pass
        print(f"[mem:{tag}] " + " | ".join(parts), flush=True)

    # ---- background watchdog: only way to see live cache/RAM growth ------
    # Jupyter/Kaggle runs one cell at a time; you cannot open a second cell
    # to `du -sh` the cache dir while this one is busy. This thread prints
    # into THIS cell's own output on a timer instead, so cache-growth and
    # RAM-growth are visible live without a separate terminal/cell.
    if args.watchdog_interval > 0:
        def _watchdog():
            while True:
                time.sleep(args.watchdog_interval)
                try:
                    files = [p for p in Path(args.xla_cache_dir).rglob("*") if p.is_file()]
                    cache_gb = sum(p.stat().st_size for p in files) / 1024**3
                except Exception as e:
                    files, cache_gb = [], -1
                mem_total = mem_avail = None
                try:
                    with open("/proc/meminfo") as f:
                        info = {ln.split(":")[0]: ln.split()[1] for ln in f}
                    mem_total = int(info["MemTotal"]) / 1024**2
                    mem_avail = int(info["MemAvailable"]) / 1024**2
                except Exception:
                    pass
                shm_gb = None
                try:
                    st = os.statvfs("/dev/shm")
                    shm_gb = (st.f_blocks - st.f_bfree) * st.f_frsize / 1024**3
                except Exception:
                    pass
                msg = f"[watchdog] cache: {len(files)} files, {cache_gb:.2f}GB"
                try:
                    hit = met.counter_value('PersistentCacheHit') or 0
                    miss = met.counter_value('PersistentCacheMiss') or 0
                    msg += f" (hit={hit} miss={miss})"
                except Exception:
                    pass
                if shm_gb is not None:
                    msg += f" | /dev/shm used: {shm_gb:.2f}GB"
                if mem_total is not None:
                    msg += f" | RAM: {mem_total - mem_avail:.1f}/{mem_total:.1f}GB"
                print(msg, flush=True)
        threading.Thread(target=_watchdog, daemon=True).start()
        print(f"[watchdog] started, logging every {args.watchdog_interval}s "
              f"(this replaces needing a second parallel cell)", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_dir = args.checkpoint_dir
    base_dir = args.base_dir or os.path.join(checkpoint_dir, '..', 'LongCat-Video')
    model_type = "avatar-v1.5"

    # v1.5 constants (mirrors run_demo_avatar_single_audio_to_video.py)
    save_fps = args.save_fps          # 25
    audio_stride = 1
    num_frames = 93
    num_cond_frames = 13
    num_segments_auto = args.num_segments.lower() == 'auto'
    num_segments = 1 if num_segments_auto else max(1, int(args.num_segments))
    num_inference_steps = args.num_inference_steps
    text_guidance_scale = 1.0         # distill => CFG off => 1 forward per step
    audio_guidance_scale = 1.0
    use_distill = True

    if args.resolution == '480p':
        height, width = 480, 832
    else:
        height, width = 768, 1280
        print("[warn] 720p: ~2.6x more tokens => ~7x attention FLOPs and a separate "
              "compilation; try 480p first.")

    with open(args.input_json, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    prompt = input_data['prompt']
    negative_prompt = (
        "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, "
        "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
        "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
        "disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, "
        "many people in the background, walking backwards"
    )
    raw_speech_path = input_data['cond_audio']['person1']
    raw_speech_path2 = input_data['cond_audio'].get('person2', None)
    multi_mode = raw_speech_path2 is not None
    audio_type = input_data.get('audio_type', 'para')
    bbox_cfg = input_data.get('bbox', None)
    if multi_mode:
        assert args.stage_1 == 'ai2v', "multitalk (dual-speaker) requires --stage_1 ai2v"
        print(f"[multitalk] dual-speaker mode: audio_type={audio_type} "
              f"bbox={'yes' if bbox_cfg else 'auto left/right split'}", flush=True)

    from transformers import AutoTokenizer, UMT5EncoderModel
    from diffusers.utils import load_image

    from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.modules.xla_loading import load_dit_xla_spmd
    from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
    from longcat_video.audio_process.torch_utils import save_video_ffmpeg

    # CPU generator: noise is sampled on host and transferred (deterministic on XLA)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    # =====================================================================
    # STAGE 0 — small components; VAE stays on HOST CPU in fp32
    # =====================================================================
    print("[stage 0] tokenizer / scheduler / VAE (cpu, fp32)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_dir, subfolder="tokenizer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler")

    if args.runtime_scalar_fix:
        # === RUNTIME_SCALAR_FIX ===
        # Root cause of the ~100 s/step timings: every denoise step compiled a
        # DIFFERENT XLA graph, because
        #   * scheduler.timesteps lived on the XLA device, so
        #     `for i, t in enumerate(timesteps)` baked a different select-index
        #     constant into each step's graph, and
        #   * scheduler.step() indexed sigmas with the python int step_index
        #     (scheduling_flow_match_euler_discrete.py:448-454), baking per-step
        #     sigma/dt constants into the HLO.
        # Fix: pin the schedule to the host and transfer the per-step scalars
        # (timestep tensor, sigma/dt) to the device as graph *inputs*
        # (transferred tensors become HLO parameters, not inline constants).
        # All denoise steps of a given shape then share a single compiled graph.
        from longcat_video.modules.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteSchedulerOutput,
        )

        _orig_set_timesteps = scheduler.set_timesteps

        def _set_timesteps_on_host(*sa, **skw):
            skw["device"] = "cpu"  # keep timesteps/sigmas host-resident
            return _orig_set_timesteps(*sa, **skw)

        scheduler.set_timesteps = _set_timesteps_on_host

        _orig_sched_step = scheduler.step

        def _sched_step_runtime_scalars(model_output, timestep, sample,
                                        per_token_timesteps=None, return_dict=True, **skw):
            if per_token_timesteps is not None:  # not used by this runner; keep original path
                return _orig_sched_step(model_output, timestep, sample,
                                        per_token_timesteps=per_token_timesteps,
                                        return_dict=return_dict, **skw)
            if scheduler.step_index is None:
                scheduler._init_step_index(timestep)  # host-side .item(), no device sync
            idx = scheduler.step_index
            sigma = float(scheduler.sigmas[idx])
            sigma_next = float(scheduler.sigmas[idx + 1])
            dev = model_output.device
            sample = sample.to(torch.float32)
            if scheduler.config.stochastic_sampling:
                cur = torch.tensor(sigma, dtype=torch.float32).to(dev)
                nxt = torch.tensor(sigma_next, dtype=torch.float32).to(dev)
                x0 = sample - cur * model_output
                noise = torch.randn_like(sample)
                prev_sample = (1.0 - nxt) * x0 + nxt * noise
            else:
                dt = torch.tensor(sigma_next - sigma, dtype=torch.float32).to(dev)
                prev_sample = sample + dt * model_output
            scheduler._step_index += 1
            prev_sample = prev_sample.to(model_output.dtype)
            if not return_dict:
                return (prev_sample,)
            return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

        scheduler.step = _sched_step_runtime_scalars
        print("[runtime-scalar-fix] scheduler schedule pinned to host; "
              "per-step timestep/sigma/dt enter the graph as transferred inputs", flush=True)

    vae_load_dtype = torch.bfloat16 if args.vae_on_tpu and args.vae_dtype == 'bf16' else torch.float32
    vae = AutoencoderKLWan.from_pretrained(
        base_dir, subfolder="vae", torch_dtype=vae_load_dtype, low_cpu_mem_usage=True
    ).eval()

    if args.vae_on_tpu:
        # === VAE ON TPU ===
        # The stock _encode/_decode loops do `out = torch.cat([out, out_], 2)`
        # every iteration, so `out` grows each frame -> a DIFFERENT XLA graph
        # per frame (that shape drift, not the feat_cache control flow, is the
        # actual recompile poison). Rewritten below: collect chunks in a list,
        # cat ONCE at the end, torch_xla.sync() per frame to keep the graphs
        # small and identical -> 2-3 compiled graphs total (first_chunk vs
        # steady state), all reusable from the persistent cache.
        import types
        from longcat_video.modules.autoencoder_kl_wan import patchify, unpatchify, DecoderOutput

        def _encode_xla(self, x):
            _, _, num_frame, height, width = x.shape
            print(f"    [vae-encode] start frames={num_frame} shape={tuple(x.shape)}", flush=True)
            t0 = _time.time()
            self.clear_cache()
            if self.config.patch_size is not None:
                x = patchify(x, patch_size=self.config.patch_size)
            iter_ = 1 + (num_frame - 1) // 4
            outs = []
            for i in range(iter_):
                step_t0 = _time.time()
                self._enc_conv_idx = [0]
                if i == 0:
                    chunk = x[:, :, :1]
                else:
                    # RUNTIME_SCALAR_FIX for the frame loop: a python-int slice
                    # start bakes a different constant into each chunk's HLO ->
                    # one compiled graph PER CHUNK. index_select with a
                    # transferred index tensor makes the position a graph
                    # *input*, so every chunk shares one graph.
                    start = 1 + 4 * (i - 1)
                    idx = torch.arange(start, start + 4, dtype=torch.long).to(x.device)
                    chunk = x.index_select(2, idx)
                outs.append(self.encoder(chunk, feat_cache=self._enc_feat_map,
                                         feat_idx=self._enc_conv_idx))
                torch_xla.sync()
                cache_delta(f"vae-encode chunk {i}")
                print(f"    [vae-encode] chunk {i + 1}/{iter_} sync {_time.time() - step_t0:.1f}s "
                      f"({_time.time() - t0:.1f}s elapsed)", flush=True)
            enc = self.quant_conv(torch.cat(outs, 2))
            torch_xla.sync()
            print(f"    [vae-encode] quant_conv done ({_time.time() - t0:.1f}s total)", flush=True)
            self.clear_cache()
            return enc

        def _tiled_decode_xla(self, z):
            # Graph-stable port of AutoencoderKLWan.tiled_decode: a full-frame
            # decoder program needs ~3.5 GB/chip which doesn't fit next to the
            # DiT shards, so decode fixed-size 256px tiles (identical shapes ->
            # identical graphs, cached) and blend the overlaps like the stock
            # implementation. sync per decoder call keeps live graphs small.
            _, _, num_frames, height, width = z.shape
            sample_height = height * self.spatial_compression_ratio
            sample_width = width * self.spatial_compression_ratio
            tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
            tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
            tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
            tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio
            blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
            blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

            n_rows = len(range(0, height, tile_latent_stride_height))
            n_cols = len(range(0, width, tile_latent_stride_width))
            t0 = _time.time()
            rows = []
            for ri, i in enumerate(range(0, height, tile_latent_stride_height)):
                row = []
                for ci, j in enumerate(range(0, width, tile_latent_stride_width)):
                    self.clear_cache()
                    time = []
                    for k in range(num_frames):
                        self._conv_idx = [0]
                        tile = z[:, :, k:k + 1, i:i + tile_latent_min_height, j:j + tile_latent_min_width]
                        tile = self.post_quant_conv(tile)
                        time.append(self.decoder(tile, feat_cache=self._feat_map,
                                                 feat_idx=self._conv_idx))
                        torch_xla.sync()
                    row.append(torch.cat(time, dim=2))
                    torch_xla.sync()
                    print(f"    [vae-decode] tile {ri * n_cols + ci + 1}/{n_rows * n_cols} "
                          f"({_time.time() - t0:.1f}s elapsed)", flush=True)
                rows.append(row)
            self.clear_cache()

            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    if i > 0:
                        tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                    if j > 0:
                        tile = self.blend_h(row[j - 1], tile, blend_width)
                    result_row.append(tile[:, :, :, :self.tile_sample_stride_height, :self.tile_sample_stride_width])
                result_rows.append(torch.cat(result_row, dim=-1))
            dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]
            torch_xla.sync()
            return dec

        def _decode_xla(self, z, return_dict=True):
            _, _, num_frame, height, width = z.shape
            print(f"    [vae-decode] start frames={num_frame} latent_shape={tuple(z.shape)} "
                  f"mode={'tiled' if args.vae_tiled_decode else 'full-frame'} dtype={z.dtype}", flush=True)
            t0 = _time.time()
            tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
            tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
            if args.vae_tiled_decode and (height > tile_latent_min_height or width > tile_latent_min_width):
                out = _tiled_decode_xla(self, z)
            else:
                self.clear_cache()
                x = self.post_quant_conv(z)
                orig_latent_w = x.shape[-1]
                pad_w = 0
                if args.vae_spatial_shard:
                    # Replicated decode = single-chip throughput. Shard the conv
                    # work spatially instead: edge-pad W to a multiple of the mesh
                    # size (GSPMD wants even shards), annotate the W axis, and let
                    # sharding propagation partition every decoder conv (halo
                    # exchanges inserted automatically). Cropped back after decode.
                    pad_w = (-orig_latent_w) % num_devices
                    if pad_w:
                        edge = x[..., -1:].expand(*x.shape[:-1], pad_w)
                        x = torch.cat([x, edge], dim=-1)
                    xs.mark_sharding(x, mesh, (None, None, None, None, 'model'))
                    print(f"    [vae-decode] W-sharded over {num_devices} chips "
                          f"(latent W {orig_latent_w}->{x.shape[-1]})", flush=True)
                torch_xla.sync()
                print(f"    [vae-decode] post_quant_conv sync {_time.time() - t0:.1f}s", flush=True)
                outs = []
                for i in range(num_frame):
                    step_t0 = _time.time()
                    self._conv_idx = [0]
                    if i == 0:
                        outs.append(self.decoder(x[:, :, :1], feat_cache=self._feat_map,
                                                 feat_idx=self._conv_idx, first_chunk=True))
                    else:
                        # RUNTIME_SCALAR_FIX for the frame loop: x[:, :, i:i+1]
                        # bakes the slice start i into the HLO as a constant ->
                        # 24 frames = 24 distinct graphs = 24 cache misses (the
                        # exact bug the scheduler runtime_scalar_fix solved for
                        # denoise steps). index_select with a transferred index
                        # tensor turns the position into a graph *input*: all
                        # steady-state frames share ONE graph (frames 1-2 still
                        # differ while the temporal feat_cache builds up).
                        idx = torch.tensor([i], dtype=torch.long).to(x.device)
                        frame = x.index_select(2, idx)
                        outs.append(self.decoder(frame, feat_cache=self._feat_map,
                                                 feat_idx=self._conv_idx))
                    torch_xla.sync()
                    cache_delta(f"vae-decode frame {i}")
                    print(f"    [vae-decode] frame {i + 1}/{num_frame} sync {_time.time() - step_t0:.1f}s "
                          f"({_time.time() - t0:.1f}s elapsed)", flush=True)
                out = torch.cat(outs, 2)
                if pad_w:
                    up = out.shape[-1] // x.shape[-1]
                    out = out[..., :orig_latent_w * up]
            if self.config.patch_size is not None:
                out = unpatchify(out, patch_size=self.config.patch_size)
            out = torch.clamp(out, min=-1.0, max=1.0)
            torch_xla.sync()
            print(f"    [vae-decode] final clamp/sync done ({_time.time() - t0:.1f}s total)", flush=True)
            self.clear_cache()
            if not return_dict:
                return (out,)
            return DecoderOutput(sample=out)

        vae._encode = types.MethodType(_encode_xla, vae)
        vae._decode = types.MethodType(_decode_xla, vae)
        vae = vae.to(device=xla_dev, dtype=vae_load_dtype)
        print(f"[vae-on-tpu] Wan VAE moved to XLA ({vae_load_dtype}, replicated); "
              f"decode={'tiled' if args.vae_tiled_decode else 'full-frame'}; "
              "frame loops patched to graph-stable list-collect + per-frame sync", flush=True)
    log_mem("vae")

    # =====================================================================
    # STAGE 1 — vocal separation + Whisper audio embedding, all on CPU
    # =====================================================================
    print("[stage 1] vocal separation + Whisper-large-v3 audio embedding (cpu)", flush=True)
    from audio_separator.separator import Separator
    import librosa

    audio_tmp_dir = Path("./audio_temp_file")
    (audio_tmp_dir / "vocals").mkdir(parents=True, exist_ok=True)
    vocal_separator_path = os.path.join(checkpoint_dir, 'vocal_separator', 'Kim_Vocal_2.onnx')
    vocal_separator = Separator(
        output_dir=audio_tmp_dir / "vocals",
        output_single_stem="vocals",
        model_file_dir=os.path.dirname(vocal_separator_path),
    )
    vocal_separator.load_model(os.path.basename(vocal_separator_path))
    temp_vocal_path = extract_vocal_from_speech(
        raw_speech_path, f"/tmp/temp_speech_{generate_random_uid()}_vocal.wav",
        vocal_separator, audio_tmp_dir,
    )
    assert temp_vocal_path is not None and os.path.exists(temp_vocal_path), "No vocal detected"

    sr = 16000
    temp_vocal_paths = [temp_vocal_path]
    if multi_mode:
        temp_vocal_path2 = extract_vocal_from_speech(
            raw_speech_path2, f"/tmp/temp_speech_{generate_random_uid()}_vocal2.wav",
            vocal_separator, audio_tmp_dir,
        )
        assert temp_vocal_path2 is not None and os.path.exists(temp_vocal_path2), "No vocal detected in person2 audio"
        temp_vocal_paths.append(temp_vocal_path2)
        from gradio_server import prepare_multi_audio
        left_arr, right_arr, merged_raw = prepare_multi_audio(
            temp_vocal_path, temp_vocal_path2, raw_speech_path, raw_speech_path2,
            sr=sr, audio_type=audio_type,
        )
        person_speech_arrays = [left_arr, right_arr]
        import soundfile as sf
        mux_audio_path = os.path.join(args.output_dir, "merged_audio.wav")
        sf.write(mux_audio_path, merged_raw, sr)
        source_duration = len(left_arr) / sr
    else:
        speech_array, sr = librosa.load(temp_vocal_path, sr=sr)
        person_speech_arrays = [speech_array]
        mux_audio_path = raw_speech_path
        source_duration = len(person_speech_arrays[0]) / sr
    del vocal_separator
    gc.collect()

    if num_segments_auto:
        # generate_duration = num_frames/fps + (N-1)*(num_frames-num_cond_frames)/fps >= source_duration
        # solve for N:
        if source_duration * save_fps <= num_frames:
            num_segments = 1
        else:
            num_segments = max(1, math.ceil(
                1 + (source_duration * save_fps - num_frames) / (num_frames - num_cond_frames)))
        print(f"    [auto] audio {source_duration:.1f}s -> {num_segments} segment(s)", flush=True)

    generate_duration = num_frames / save_fps + (num_segments - 1) * (num_frames - num_cond_frames) / save_fps
    added = math.ceil((generate_duration - source_duration) * sr)
    if added > 0:
        person_speech_arrays = [np.append(a, [0.] * added) for a in person_speech_arrays]
    print(f"    audio {source_duration:.1f}s, target video {generate_duration:.1f}s "
          f"({num_segments} segment(s), {num_frames + (num_segments-1)*(num_frames-num_cond_frames)} frames)")

    audio_model_path = os.path.join(checkpoint_dir, 'whisper-large-v3')
    audio_encoder = get_audio_encoder(audio_model_path, model_type).float().eval()  # CPU fp32
    audio_feature_extractor = get_audio_feature_extractor(audio_model_path, model_type)

    pipe = LongCatVideoAvatarPipeline(
        tokenizer=tokenizer, text_encoder=None, vae=vae, scheduler=scheduler,
        dit=None, audio_encoder=audio_encoder,
        audio_feature_extractor=audio_feature_extractor, model_type=model_type,
    )
    pipe.device = xla_dev  # DiT device; VAE bridging is handled inside the pipeline

    full_audio_embs = []
    for pi, arr in enumerate(person_speech_arrays):
        with torch.no_grad():
            emb = pipe.get_audio_embedding(
                arr, fps=save_fps * audio_stride, device="cpu",
                sample_rate=sr, model_type=model_type,
            )
        if torch.isnan(emb).any():
            raise ValueError(f"broken audio embedding (person{pi+1}) with nan values")
        full_audio_embs.append(emb.float().cpu())
        print(f"    audio embedding person{pi+1}: {tuple(emb.shape)}")
    if multi_mode:
        assert full_audio_embs[0].shape == full_audio_embs[1].shape, "Inconsistent audio embedding shape"

    pipe.audio_encoder = None
    del audio_encoder
    for p in temp_vocal_paths:
        if os.path.exists(p):
            os.remove(p)
    torch_gc()
    log_mem("audio done")

    # =====================================================================
    # STAGE 2 — umT5-xxl text encoder on CPU -> embeddings cached -> freed
    # =====================================================================
    print("[stage 2] umT5-xxl text encoding (cpu)", flush=True)
    try:
        import psutil
        te_dtype = torch.float32 if psutil.virtual_memory().available > 80 * 2**30 else torch.bfloat16
    except Exception:
        te_dtype = torch.bfloat16
    text_encoder = UMT5EncoderModel.from_pretrained(
        base_dir, subfolder="text_encoder", torch_dtype=te_dtype, low_cpu_mem_usage=True,
    ).eval()
    pipe.text_encoder = text_encoder
    log_mem("text encoder loaded")

    with torch.no_grad():
        pe, pm, npe, npm = pipe.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,  # cache both branches; distill uses the positive one
            num_videos_per_prompt=1, max_sequence_length=512,
            dtype=torch.bfloat16, device=torch.device("cpu"),
        )
    pipe.set_cached_text_embeddings(pe.cpu(), pm.cpu(), npe.cpu(), npm.cpu())

    pipe.text_encoder = None
    del text_encoder
    torch_gc()
    log_mem("text done")

    # =====================================================================
    # STAGE 3 — bf16 DiT: LoRA merged on host, sharded over the 8-chip mesh
    # =====================================================================
    print("[stage 3] loading bf16 DiT, merging DMD LoRA, sharding over the mesh", flush=True)
    t0 = time.time()
    distill_ckpt = os.path.join(checkpoint_dir, 'lora', 'dmd_lora.safetensors')
    assert os.path.exists(distill_ckpt), f"missing DMD LoRA: {distill_ckpt}"
    dit = load_dit_xla_spmd(
        checkpoint_dir, mesh,
        subfolder="base_model", dtype=torch.bfloat16,
        lora_path=distill_ckpt, lora_multiplier=1.0, lora_dim=128, lora_alpha=64,
        cp_split_hw=[1, 1],
    ).eval()
    pipe.dit = dit
    torch_gc()
    print(f"    dit ready in {(time.time()-t0)/60:.1f} min")
    log_mem("dit loaded")

    # === DIT_XLA_MIXED_PRECISION_PATCH ===
    # Fix XLA "mixed precision is disallowed" from Linear lowering:
    #   dot may produce f32, while Linear.bias is bf16.
    # Do not use global XLA_DOWNCAST_BF16 because CPU Whisper must stay fp32.
    print("[dtype-fix] installing DiT/XLA mixed-precision safe Linear patch", flush=True)

    import torch.nn.functional as F

    # The loader already requested bf16, but keep this to normalize any LoRA-merged
    # residual parameters/buffers. This is scoped to DiT only.
    pipe.dit.to(dtype=torch.bfloat16)

    def _to_bf16_tree_for_dit(x):
        if torch.is_tensor(x):
            if args.runtime_scalar_fix and x.device.type != "xla":
                # host-resident per-step tensors (e.g. the timestep from the
                # host-pinned scheduler) enter the graph as transferred inputs,
                # not baked constants -> no per-step recompilation
                x = x.to(xla_dev)
            if x.is_floating_point():
                return x.to(dtype=torch.bfloat16)
            return x
        if isinstance(x, tuple):
            return tuple(_to_bf16_tree_for_dit(v) for v in x)
        if isinstance(x, list):
            return [_to_bf16_tree_for_dit(v) for v in x]
        if isinstance(x, dict):
            return {k: _to_bf16_tree_for_dit(v) for k, v in x.items()}
        return x

    if not hasattr(pipe.dit, "_orig_forward_xla_dtype_patch"):
        pipe.dit._orig_forward_xla_dtype_patch = pipe.dit.forward

        def _dit_forward_cast_inputs(*args, **kwargs):
            # === GRAPH-BOUNDARY CUT ===
            # Lazy tracing folds every pending op (VAE cond-encode tail, latent
            # prep, dtype casts) into the NEXT synced graph, i.e. denoise step 0.
            # Any VAE/prep change then invalidates the cached step-0 graph even
            # though the DiT is untouched. Cutting here, before the first DiT
            # forward of each generation (step_index is still None), isolates
            # prep into its own tiny graph so the expensive denoise graphs stay
            # cache-stable across VAE/prep code changes.
            if getattr(scheduler, "_step_index", 0) is None:
                torch_xla.sync()
            # Cast only incoming floating tensors to bf16.
            # Do NOT cast the DiT output here: the model intentionally returns fp32.
            args = _to_bf16_tree_for_dit(args)
            kwargs = _to_bf16_tree_for_dit(kwargs)
            return pipe.dit._orig_forward_xla_dtype_patch(*args, **kwargs)

        pipe.dit.forward = _dit_forward_cast_inputs

    _bad = []
    for _name, _t in list(pipe.dit.named_parameters()) + list(pipe.dit.named_buffers()):
        if hasattr(_t, "dtype") and _t.is_floating_point() and _t.dtype != torch.bfloat16:
            _bad.append((_name, str(_t.dtype), tuple(_t.shape)))
            if len(_bad) >= 20:
                break
    print("[dtype-fix] non-bf16 DiT tensors shown:", _bad, flush=True)

    # Candidate locator for the current failing shape f32[24,512].
    # Expected high-value candidates:
    #   t_embedder.mlp.0 / t_embedder.mlp.2
    #   audio_proj.proj1 / proj1_vf / proj2
    _linear_512 = []
    for _name, _m in pipe.dit.named_modules():
        if isinstance(_m, torch.nn.Linear) and getattr(_m, "out_features", None) == 512:
            _linear_512.append((
                _name,
                int(_m.in_features),
                int(_m.out_features),
                str(_m.weight.dtype),
                None if _m.bias is None else str(_m.bias.dtype),
            ))
    print("[dtype-fix] Linear out_features=512 candidates/count:", len(_linear_512), _linear_512[:20], flush=True)

    def _patch_linear_xla_safe(_name, _m):
        if not isinstance(_m, torch.nn.Linear):
            return 0
        if getattr(_m, "_xla_mixed_precision_safe", False):
            return 0

        _m._orig_forward_xla_mixed_precision_safe = _m.forward
        _m._xla_debug_name = _name
        _m._xla_mixed_precision_safe = True

        def _linear_forward_xla_safe(x, _m=_m):
            if torch.is_tensor(x) and x.is_floating_point() and x.device.type == "xla":
                # Avoid F.linear(..., bias) because XLA can lower it as:
                #   f32 dot + bf16 broadcast(bias)
                # which fails before optimization when mixed precision is disallowed.
                out = F.linear(x, _m.weight, None)

                if _m.bias is not None:
                    out = out + _m.bias.to(dtype=out.dtype)

                # Preserve fp32 control/modulation streams:
                #   TimestepEmbedder and AdaLN modulation intentionally run from fp32 t.
                # For normal activation streams, cast Linear output back to input dtype
                # so later residual adds stay bf16 + bf16.
                if x.dtype != torch.float32 and out.dtype != x.dtype:
                    out = out.to(dtype=x.dtype)

                return out

            return _m._orig_forward_xla_mixed_precision_safe(x)

        _m.forward = _linear_forward_xla_safe
        return 1

    _linear_patch_count = 0
    for _name, _m in pipe.dit.named_modules():
        _linear_patch_count += _patch_linear_xla_safe(_name, _m)

    print("[dtype-fix] patched Linear modules:", _linear_patch_count, flush=True)


    # ---- graph cut BEFORE every DiT forward ----
    # XLA is lazy: without this, all pre-loop prep ops (latent init, condition
    # concat, masks, audio-emb processing) stay pending and get fused INTO
    # step 0's graph -> step 0 compiles a ~95s giant that differs from the
    # steady-state step graph, which then compiles AGAIN for another ~95s at
    # step 1 (observed: step0 97.9s + step1 95.5s, both misses). Cutting right
    # before the forward flushes the prep into its own small graph, so step 0
    # traces the SAME graph steps 1..N-1 use: one big compile instead of two.
    # In steady state the pre-forward pending set is tiny (timestep expansion,
    # model-input concat) and compiles once, so the extra sync is ~free. Only
    # safe when we sync every step anyway; lazy multi-step mode keeps the old
    # behavior.
    if args.sync_every_n_steps == 1:
        _orig_dit_forward = pipe.dit.forward

        def _dit_forward_precut(*fa, **fkw):
            torch_xla.sync()
            cache_delta("pre-forward prep")
            return _orig_dit_forward(*fa, **fkw)

        pipe.dit.forward = _dit_forward_precut
        print("[graph-cut] torch_xla.sync() inserted before every DiT forward "
              "(step-0 prep no longer fused into the denoise graph)", flush=True)

    # ---- graph cut + timing after every denoise step ----
    step_state = {"t": None}
    def _after_step(i):
        # XLA is lazy: sync() is the graph-cut point. Syncing every step is
        # safest for HBM; a larger cadence reduces compile/execute barriers.
        do_sync = ((i + 1) % max(1, args.sync_every_n_steps) == 0) or (i + 1 == num_inference_steps)
        if do_sync:
            torch_xla.sync()
            xm.wait_device_ops()
            cache_delta(f"denoise step {i}")
        now = time.time()
        if step_state["t"] is not None:
            tag = "sync" if do_sync else "lazy"
            print(f"    [step {i} {tag}] {now - step_state['t']:.1f}s", flush=True)
        step_state["t"] = now
    pipe._after_denoise_step = _after_step

    # ---- attribute AVC KV-cache-build compiles ----
    # With offload_kv_cache=True every block's KV is transferred to host,
    # which cuts the trace into many small per-block graphs. Their first-time
    # compiles show up in the watchdog as anonymous cache-file growth; this
    # wrapper brackets the build with syncs so the misses are labeled and the
    # phase is timed. Purely observability, no behavior change.
    _orig_cache_clean = pipe._cache_clean_latents

    def _cache_clean_latents_tagged(*ca, **ckw):
        torch_xla.sync()
        cache_delta("avc kv-build: pre")
        t0 = time.time()
        out = _orig_cache_clean(*ca, **ckw)
        torch_xla.sync()
        xm.wait_device_ops()
        cache_delta("avc kv-build")
        print(f"    [avc kv-build] {time.time() - t0:.1f}s "
              "(one-time compiles land here; cached for later segments/runs)", flush=True)
        return out

    pipe._cache_clean_latents = _cache_clean_latents_tagged

    # ---- optional profiler: trace segment-1 denoising ----
    if args.profile:
        import torch_xla.debug.profiler as xp
        os.makedirs(args.profile_dir, exist_ok=True)
        _server = xp.start_server(9012)  # noqa: F841  must stay referenced

        def _capture():
            # wait past the first (compilation-dominated) steps, then trace
            time.sleep(60)
            print(f"[profile] capturing {args.profile_duration_ms} ms -> {args.profile_dir}", flush=True)
            try:
                xp.trace_detached("localhost:9012", args.profile_dir,
                                  duration_ms=args.profile_duration_ms)
                print("[profile] trace saved", flush=True)
            except Exception as e:
                print(f"[profile] trace failed: {e}", flush=True)
        threading.Thread(target=_capture, daemon=True).start()

    # =====================================================================
    # STAGE 4 — generation
    # =====================================================================
    indices = torch.arange(2 * 2 + 1) - 2
    audio_start_idx = 0
    audio_end_idx = audio_start_idx + audio_stride * num_frames

    # Single: [1, T, ...]; multitalk: [2, T, ...] (person1 + person2 stacked on batch)
    def slice_audio_emb(start, end):
        ci = torch.arange(start, end, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
        ci = torch.clamp(ci, min=0, max=full_audio_embs[0].shape[0] - 1)
        return torch.cat(
            [emb[ci][None, ...] for emb in full_audio_embs]
        ).to(device=xla_dev, dtype=torch.bfloat16)

    # --- ref_target_masks (multitalk): [3, H, W] = person1/person2/background ---
    ref_target_masks = None
    if multi_mode:
        from gradio_server import build_ref_target_masks
        cond_img = PIL.Image.open(input_data['cond_image'])
        src_w, src_h = cond_img.size
        bbox_p1 = bbox_p2 = None
        if bbox_cfg:
            # official json bbox order: [y_min, x_min, y_max, x_max]
            b1 = bbox_cfg.get('person1')
            b2 = bbox_cfg.get('person2')
            if b1 and b2:
                bbox_p1 = [b1[1], b1[0], b1[3], b1[2]]  # -> [x1, y1, x2, y2]
                bbox_p2 = [b2[1], b2[0], b2[3], b2[2]]
        ref_target_masks = build_ref_target_masks(bbox_p1, bbox_p2, src_w, src_h)
        ref_target_masks = ref_target_masks.to(device=xla_dev, dtype=torch.float32)
        print(f"[multitalk] ref_target_masks: {tuple(ref_target_masks.shape)}", flush=True)

    # =====================================================================
    # WARMUP — compile every graph on throwaway data, kept OUT of the timed
    # stage-4 run below. 2 steps is enough: step 0 and the steady-state graph
    # steps 1-7 all reuse are the only two DISTINCT denoise graphs (see the
    # step-timing log: step2..7 are already identical to step1). Real
    # segment timings after this block are therefore pure execution time,
    # never a mix of compile + compute.
    # =====================================================================
    if args.warmup:
        print("[warmup] compiling ai2v/avc + VAE graphs on a throwaway 2-step run ...", flush=True)
        t_warmup = time.time()
        # avatar-v1.5 distill: get_timesteps_sigmas ignores sampling_steps and
        # always returns num_distill_sample_steps (8) sigmas; passing a smaller
        # warmup_steps causes a len(sigmas) != num_inference_steps mismatch.
        # Use the full step count; the graph for step 2-7 is compiled on the
        # first steady-state step anyway so no compile savings from truncating.
        warmup_steps = num_inference_steps
        warmup_generator = torch.Generator(device="cpu").manual_seed(0)
        warmup_audio_emb = slice_audio_emb(0, audio_stride * num_frames)
        common_warmup = dict(
            prompt=prompt, negative_prompt=negative_prompt,
            num_frames=num_frames, num_inference_steps=warmup_steps,
            text_guidance_scale=text_guidance_scale, audio_guidance_scale=audio_guidance_scale,
            generator=warmup_generator, output_type='both', audio_emb=warmup_audio_emb, use_distill=use_distill,
        )
        if args.stage_1 == 'ai2v':
            common_warmup['ref_target_masks'] = ref_target_masks
        with torch.no_grad():
            if args.stage_1 == 'at2v':
                warm_output, warm_latent = pipe.generate_at2v(height=height, width=width, **common_warmup)
            else:
                warm_image = load_image(input_data['cond_image'])
                warm_output, warm_latent = pipe.generate_ai2v(image=warm_image, resolution=args.resolution, **common_warmup)
        warm_output = warm_output[0]
        warm_video = [PIL.Image.fromarray((warm_output[i] * 255).astype(np.uint8)) for i in range(warm_output.shape[0])]
        del warm_output
        torch_gc()

        if num_segments > 1:
            warm_ref_latent = warm_latent[:, :, :1].clone()
            with torch.no_grad():
                pipe.generate_avc(
                    video=warm_video, video_latent=warm_latent,
                    prompt=prompt, negative_prompt=negative_prompt,
                    height=warm_video[0].size[1], width=warm_video[0].size[0],
                    num_frames=num_frames, num_cond_frames=num_cond_frames,
                    num_inference_steps=warmup_steps,
                    text_guidance_scale=text_guidance_scale, audio_guidance_scale=audio_guidance_scale,
                    generator=warmup_generator, output_type='latent',
                    use_kv_cache=True, offload_kv_cache=args.offload_kv_cache,
                    enhance_hf=False,
                    audio_emb=warmup_audio_emb, ref_latent=warm_ref_latent,
                    ref_img_index=args.ref_img_index, mask_frame_range=args.mask_frame_range,
                    use_distill=use_distill, ref_target_masks=ref_target_masks,
                )
            del warm_ref_latent

        del warm_latent, warm_video
        pipe._clear_cache()
        torch_gc()
        torch_xla.sync()
        xm.wait_device_ops()
        log_xla_cache_stats("warmup")
        print(f"[warmup] done in {(time.time()-t_warmup)/60:.1f} min "
              "(compile cost absorbed here; segment timings below are clean execution time)", flush=True)

    # ---- resume support -------------------------------------------------
    resume_path = os.path.join(args.output_dir, "resume_state.pt")
    resume_state = None
    if args.resume and os.path.exists(resume_path):
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        print(f"[resume] loaded {resume_path}: "
              f"{resume_state['segments_done']}/{num_segments} segments already done", flush=True)

    def _save_resume_state(segments_done, latent, ref_latent, current_video,
                           all_generated_frames, audio_start_idx):
        state = {
            "segments_done": segments_done,
            "latent": latent.detach().cpu(),
            "ref_latent": ref_latent.detach().cpu(),
            "current_video": np.array(current_video),
            "all_generated_frames": np.array(all_generated_frames),
            "audio_start_idx": audio_start_idx,
            "generator_state": generator.get_state(),
        }
        tmp = resume_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, resume_path)  # atomic: never leave a half-written ckpt
        print(f"[resume] checkpoint saved ({segments_done}/{num_segments} segments)", flush=True)

    if resume_state is None:
        audio_emb = slice_audio_emb(audio_start_idx, audio_end_idx)

        print(f"[stage 4] segment 1/{num_segments} ({args.stage_1}) ...", flush=True)
        t0 = time.time()
        step_state["t"] = time.time()
        common = dict(
            prompt=prompt, negative_prompt=negative_prompt,
            num_frames=num_frames, num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale, audio_guidance_scale=audio_guidance_scale,
            generator=generator, output_type='both', audio_emb=audio_emb, use_distill=use_distill,
        )
        if args.stage_1 == 'ai2v':
            common['ref_target_masks'] = ref_target_masks
        with torch.no_grad():
            if args.stage_1 == 'at2v':
                output, latent = pipe.generate_at2v(height=height, width=width, **common)
            else:
                image = load_image(input_data['cond_image'])
                output, latent = pipe.generate_ai2v(image=image, resolution=args.resolution, **common)

        output = output[0]
        video = [PIL.Image.fromarray((output[i] * 255).astype(np.uint8)) for i in range(output.shape[0])]
        del output
        torch_gc()

        save_video_ffmpeg(
            torch.from_numpy(np.array(video)),
            os.path.join(args.output_dir, f"{args.stage_1}_segment_1"),
            mux_audio_path, fps=save_fps, quality=5,
        )
        print(f"    segment 1 done in {(time.time()-t0)/60:.1f} min", flush=True)
        print(f"[shape] actual video size={video[0].size} latent={tuple(latent.shape)} "
              f"audio_emb={tuple(audio_emb.shape)}  <- cache bucket key, NOT '{args.resolution}'", flush=True)
        log_xla_cache_stats("segment 1")
        if args.profile:
            report = met.metrics_report()
            print("[profile] torch_xla metrics report (head):\n" + report[:4000], flush=True)
            with open(os.path.join(args.profile_dir if os.path.isdir(args.profile_dir)
                                   else args.output_dir, "xla_metrics_segment1.txt"), "w") as f:
                f.write(report)
        log_mem("segment 1")

        # ---- long-video continuation ----------------------------------
        width, height = video[0].size
        current_video = video
        ref_latent = latent[:, :, :1].clone()
        all_generated_frames = list(video)

        segments_done = 1
        if args.resume:
            _save_resume_state(segments_done, latent, ref_latent, current_video,
                               all_generated_frames, audio_start_idx)
    else:
        # restore everything the avc loop needs from the checkpoint
        segments_done = resume_state["segments_done"]
        latent = resume_state["latent"].to(xla_dev)
        ref_latent = resume_state["ref_latent"].to(xla_dev)
        current_video = [PIL.Image.fromarray(f) for f in resume_state["current_video"]]
        all_generated_frames = [PIL.Image.fromarray(f) for f in resume_state["all_generated_frames"]]
        audio_start_idx = int(resume_state["audio_start_idx"])
        generator.set_state(resume_state["generator_state"])
        width, height = current_video[0].size
        del resume_state
        print(f"[resume] skipping segments 1..{segments_done}; "
              f"continuing at segment {segments_done + 1}", flush=True)

    # Free every segment-1 device buffer we no longer reference before the
    # (bigger) avc graph loads: its per-chip runtime reservation is ~8.5G and
    # the margin on a 16G chip is thin.
    pipe._clear_cache()
    torch_gc()
    torch_xla.sync()
    xm.wait_device_ops()

    for segment_idx in range(1, num_segments):
        if segment_idx < segments_done:
            continue  # finished in a previous run (restored via --resume)
        print(f"[stage 4] segment {segment_idx+1}/{num_segments} (avc) ...", flush=True)
        t0 = time.time()
        step_state["t"] = time.time()

        audio_start_idx = audio_start_idx + audio_stride * (num_frames - num_cond_frames)
        audio_end_idx = audio_start_idx + audio_stride * num_frames
        audio_emb = slice_audio_emb(audio_start_idx, audio_end_idx)

        with torch.no_grad():
            output, latent = pipe.generate_avc(
                video=current_video, video_latent=latent,
                prompt=prompt, negative_prompt=negative_prompt,
                height=height, width=width,
                num_frames=num_frames, num_cond_frames=num_cond_frames,
                num_inference_steps=num_inference_steps,
                text_guidance_scale=text_guidance_scale, audio_guidance_scale=audio_guidance_scale,
                generator=generator, output_type='both',
                use_kv_cache=True, offload_kv_cache=args.offload_kv_cache,  # HBM is 16 GB PER CHIP; default: host
                enhance_hf=False,  # distill mode: must be off
                audio_emb=audio_emb, ref_latent=ref_latent,
                ref_img_index=args.ref_img_index, mask_frame_range=args.mask_frame_range,
                use_distill=use_distill, ref_target_masks=ref_target_masks,
            )

        output = output[0]
        new_video = [PIL.Image.fromarray((output[i] * 255).astype(np.uint8)) for i in range(output.shape[0])]
        del output

        all_generated_frames.extend(new_video[num_cond_frames:])
        current_video = new_video

        pipe._clear_cache()
        torch_gc()

        save_video_ffmpeg(
            torch.from_numpy(np.array(all_generated_frames)),
            os.path.join(args.output_dir, f"video_continue_{segment_idx+1}"),
            mux_audio_path, fps=save_fps, quality=5,
        )
        print(f"    segment {segment_idx+1} done in {(time.time()-t0)/60:.1f} min "
              f"(total {len(all_generated_frames)} frames = {len(all_generated_frames)/save_fps:.1f} s)", flush=True)
        log_xla_cache_stats(f"segment {segment_idx+1}")
        log_mem(f"segment {segment_idx+1}")

        segments_done = segment_idx + 1
        if args.resume:
            _save_resume_state(segments_done, latent, ref_latent, current_video,
                               all_generated_frames, audio_start_idx)

    print("[done] outputs in", args.output_dir, flush=True)


if __name__ == "__main__":
    main()