"""
Gradio web UI for LongCat Video Avatar generation on Kaggle.

Supports two backends:
  --device tpu   : TPU v5e-8 (SPMD, bf16 DiT, Pallas FlashAttention, XLA graph cuts)
  --device gpu   : 2× T4 GPU (INT8 DiT sharded across cuda:0/cuda:1, SDPA attention)

Usage (TPU):
    %cd /kaggle/working/longcat-tpu-2
    !python gradio_server.py --device tpu \
        --checkpoint_dir /dev/shm/weights/LongCat-Video-Avatar-1.5 \
        --base_dir /dev/shm/weights/LongCat-Video --share

Usage (GPU T4 x2):
    %cd /kaggle/working/LongCat-Video-KaggleT4
    !python gradio_server.py --device gpu \
        --checkpoint_dir /kaggle/temp/weights/LongCat-Video-Avatar-1.5 \
        --base_dir /kaggle/temp/weights/LongCat-Video --share

Or for quick UI test without accelerator:
    !python gradio_server.py --dummy
"""

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

import os
import sys
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import gc
import json
import math
import random
import shutil
import threading
import time
import time as _time
from pathlib import Path

import numpy as np
import PIL.Image
import torch

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'tpu', 'gpu'],
                        help='auto = detect from environment (TPU if torch_xla+PJRT, else CUDA)')
    p.add_argument('--checkpoint_dir', type=str, default='/dev/shm/weights/LongCat-Video-Avatar-1.5')
    p.add_argument('--base_dir', type=str, default=None)
    p.add_argument('--resolution', type=str, default='480p', choices=['480p', '720p'])
    p.add_argument('--num_inference_steps', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_fps', type=int, default=25)
    # TPU-only args
    p.add_argument('--xla_cache_dir', type=str, default='/dev/shm/xla_cache')
    p.add_argument('--vae_on_tpu', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--vae_dtype', type=str, default='bf16', choices=['bf16', 'fp32'])
    p.add_argument('--vae_spatial_shard', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--vae_tiled_decode', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--sync_every_n_steps', type=int, default=1)
    p.add_argument('--runtime_scalar_fix', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--warmup', action=argparse.BooleanOptionalAction, default=False)
    # GPU-only args
    p.add_argument('--split_index', type=int, default=22,
                        help='GPU: DiT block index for cuda:0/cuda:1 boundary')
    # Common
    p.add_argument('--offload_kv_cache', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--ref_img_index', type=int, default=10)
    p.add_argument('--mask_frame_range', type=int, default=3)
    p.add_argument('--host', type=str, default='0.0.0.0')
    p.add_argument('--port', type=int, default=7860)
    p.add_argument('--share', action='store_true', help='create a public Gradio link')
    p.add_argument('--dummy', action='store_true', help='run without accelerator (for UI testing)')
    return p.parse_args(argv)


def torch_gc():
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def generate_random_uid():
    return str(int(time.time()))[-6:] + str(random.randint(100000, 999999))


# ---------------------------------------------------------------------------
# Global state — populated by init_models(), used by generate()
# ---------------------------------------------------------------------------
class GlobalState:
    def __init__(self):
        self.ready = False
        self.dummy = False
        self.device_type = None  # 'tpu' or 'gpu'
        self.load_stage = "Initializing..."  # For UI progress display
        self.pipe = None
        self.mesh = None
        self.xla_dev = None
        self.dev0 = None
        self.dev1 = None
        self.tokenizer = None
        self.scheduler = None
        self.vae = None
        self.checkpoint_dir = None
        self.base_dir = None
        self.model_type = "avatar-v1.5"
        self.args = None
        self.vocal_separator = None
        self.num_frames = 93
        self.num_cond_frames = 13
        self.save_fps = 25
        self.audio_stride = 1
        self.negative_prompt = (
            "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, "
            "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
            "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
            "disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, "
            "many people in the background, walking backwards"
        )
        self.step_state = {"t": None}
        self.progress_cb = None  # set per-request


GS = GlobalState()


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
def detect_device(args):
    if args.device == 'tpu':
        return 'tpu'
    if args.device == 'gpu':
        return 'gpu'
    # auto: check torch_xla importable WITHOUT initializing the computation client
    # (calling xr.global_runtime_device_count() would init the client and crash
    #  when init_environment_tpu() tries to init it again)
    try:
        import importlib
        importlib.import_module("torch_xla")
        return 'tpu'
    except ImportError:
        pass
    if torch.cuda.is_available():
        return 'gpu'
    return 'tpu'  # fallback, will fail with a clear error


# ---------------------------------------------------------------------------
# TPU: environment + model init
# ---------------------------------------------------------------------------
def init_environment_tpu(args):
    GS.load_stage = "TPU environment"
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    libtpu_args = " ".join([
        "--xla_tpu_enable_async_collective_fusion=true",
        "--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true",
        "--xla_tpu_enable_async_collective_fusion_multiple_steps=true",
        "--xla_tpu_overlap_compute_collective_tc=true",
        "--xla_enable_async_all_reduce=true",
    ])
    os.environ.setdefault("LIBTPU_INIT_ARGS", libtpu_args)
    os.makedirs(args.xla_cache_dir, exist_ok=True)

    import torch_xla
    import torch_xla.runtime as xr
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs

    _cache_active = False
    try:
        import torch_xla._XLAC as _xlac
        _cache_active = _xlac._xla_computation_cache_is_initialized()
    except Exception:
        pass
    if _cache_active:
        _cache_path = os.environ.get('XLA_PERSISTENT_CACHE_PATH', '?')
        print(f"[xla-cache] persistent cache already active (path={_cache_path})", flush=True)
    else:
        try:
            xr.initialize_cache(args.xla_cache_dir, readonly=False)
            print(f"[xla-cache] persistent cache initialized: {args.xla_cache_dir}", flush=True)
        except Exception as e:
            print(f"[xla] compilation cache unavailable: {e}")

    xr.use_spmd()
    num_devices = xr.global_runtime_device_count()
    assert num_devices >= 2, f"expected a TPU slice, found {num_devices} device(s)"
    mesh = xs.Mesh(np.arange(num_devices), (num_devices,), ("model",))
    xla_dev = torch_xla.device() if hasattr(torch_xla, "device") else xm.xla_device()
    print(f"[xla] SPMD on, 1D mesh 'model' over {num_devices} devices, device={xla_dev}")

    from longcat_video import xla_utils
    xla_utils.set_global_mesh(mesh)

    GS.mesh = mesh
    GS.xla_dev = xla_dev



def init_models_tpu(args):
    """Load all models once for TPU. Takes ~5-10 min."""
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.spmd as xs

    GS.args = args
    GS.device_type = 'tpu'
    GS.checkpoint_dir = args.checkpoint_dir
    GS.base_dir = args.base_dir or os.path.join(args.checkpoint_dir, '..', 'LongCat-Video')
    GS.save_fps = args.save_fps

    checkpoint_dir = GS.checkpoint_dir
    base_dir = GS.base_dir
    model_type = GS.model_type

    GS.load_stage = "tokenizer/scheduler/VAE"
    print("[stage 0] tokenizer / scheduler / VAE (cpu, fp32)", flush=True)
    from transformers import AutoTokenizer
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan

    tokenizer = AutoTokenizer.from_pretrained(base_dir, subfolder="tokenizer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler")

    if args.runtime_scalar_fix:
        from longcat_video.modules.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteSchedulerOutput,
        )
        _orig_set_timesteps = scheduler.set_timesteps
        def _set_timesteps_on_host(*sa, **skw):
            skw["device"] = "cpu"
            return _orig_set_timesteps(*sa, **skw)
        scheduler.set_timesteps = _set_timesteps_on_host

        _orig_sched_step = scheduler.step
        def _sched_step_runtime_scalars(model_output, timestep, sample,
                                        per_token_timesteps=None, return_dict=True, **skw):
            if per_token_timesteps is not None:
                return _orig_sched_step(model_output, timestep, sample,
                                        per_token_timesteps=per_token_timesteps,
                                        return_dict=return_dict, **skw)
            if scheduler.step_index is None:
                scheduler._init_step_index(timestep)
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
        print("[runtime-scalar-fix] scheduler pinned to host", flush=True)

    vae_load_dtype = torch.bfloat16 if args.vae_on_tpu and args.vae_dtype == 'bf16' else torch.float32
    vae = AutoencoderKLWan.from_pretrained(
        base_dir, subfolder="vae", torch_dtype=vae_load_dtype, low_cpu_mem_usage=True
    ).eval()

    if args.vae_on_tpu:
        import types
        from longcat_video.modules.autoencoder_kl_wan import patchify, unpatchify
        from diffusers.utils import BaseOutput
        from dataclasses import dataclass
        @dataclass
        class DecoderOutput(BaseOutput):
            sample: torch.FloatTensor = None

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
                    start = 1 + 4 * (i - 1)
                    idx = torch.arange(start, start + 4, dtype=torch.long).to(x.device)
                    chunk = x.index_select(2, idx)
                outs.append(self.encoder(chunk, feat_cache=self._enc_feat_map,
                                         feat_idx=self._enc_conv_idx))
                torch_xla.sync()
                print(f"    [vae-encode] chunk {i + 1}/{iter_} sync {_time.time() - step_t0:.1f}s "
                      f"({_time.time() - t0:.1f}s elapsed)", flush=True)
            enc = self.quant_conv(torch.cat(outs, 2))
            torch_xla.sync()
            print(f"    [vae-encode] quant_conv done ({_time.time() - t0:.1f}s total)", flush=True)
            self.clear_cache()
            return enc

        def _decode_xla(self, z, return_dict=True):
            _, _, num_frame, height, width = z.shape
            print(f"    [vae-decode] start frames={num_frame} latent_shape={tuple(z.shape)} dtype={z.dtype}", flush=True)
            t0 = _time.time()
            self.clear_cache()
            x = self.post_quant_conv(z)
            orig_latent_w = x.shape[-1]
            pad_w = 0
            if args.vae_spatial_shard:
                num_devices = len(getattr(GS.mesh, "device_ids", [])) or 1
                pad_w = (-orig_latent_w) % num_devices
                if pad_w:
                    edge = x[..., -1:].expand(*x.shape[:-1], pad_w)
                    x = torch.cat([x, edge], dim=-1)
                xs.mark_sharding(x, GS.mesh, (None, None, None, None, 'model'))
                print(f"    [vae-decode] W-sharded over {num_devices} chips (latent W {orig_latent_w}->{x.shape[-1]})", flush=True)
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
                    idx = torch.tensor([i], dtype=torch.long).to(x.device)
                    frame = x.index_select(2, idx)
                    outs.append(self.decoder(frame, feat_cache=self._feat_map,
                                             feat_idx=self._conv_idx))
                torch_xla.sync()
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
        vae = vae.to(device=GS.xla_dev, dtype=vae_load_dtype)
        print(f"[vae-on-tpu] Wan VAE moved to XLA ({vae_load_dtype}, replicated)", flush=True)

    GS.tokenizer = tokenizer
    GS.scheduler = scheduler
    GS.vae = vae

    GS.load_stage = "text encoder"
    print("[stage 2] loading umT5-xxl text encoder (cpu)", flush=True)
    from transformers import UMT5EncoderModel
    try:
        import psutil
        te_dtype = torch.float32 if psutil.virtual_memory().available > 80 * 2**30 else torch.bfloat16
    except Exception:
        te_dtype = torch.bfloat16
    text_encoder = UMT5EncoderModel.from_pretrained(
        base_dir, subfolder="text_encoder", torch_dtype=te_dtype, low_cpu_mem_usage=True,
    ).eval()

    GS.load_stage = "DiT model"
    print("[stage 3] loading bf16 DiT, merging DMD LoRA, sharding over the mesh", flush=True)
    t0 = time.time()
    from longcat_video.modules.xla_loading import load_dit_xla_spmd
    from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
    from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor

    audio_model_path = os.path.join(checkpoint_dir, 'whisper-large-v3')
    audio_encoder = get_audio_encoder(audio_model_path, model_type).float().eval()
    audio_feature_extractor = get_audio_feature_extractor(audio_model_path, model_type)

    pipe = LongCatVideoAvatarPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=scheduler,
        dit=None, audio_encoder=audio_encoder,
        audio_feature_extractor=audio_feature_extractor, model_type=model_type,
    )
    pipe.device = GS.xla_dev

    distill_ckpt = os.path.join(checkpoint_dir, 'lora', 'dmd_lora.safetensors')
    assert os.path.exists(distill_ckpt), f"missing DMD LoRA: {distill_ckpt}"
    dit = load_dit_xla_spmd(
        checkpoint_dir, GS.mesh,
        subfolder="base_model", dtype=torch.bfloat16,
        lora_path=distill_ckpt, lora_multiplier=1.0, lora_dim=128, lora_alpha=64,
        cp_split_hw=[1, 1],
    ).eval()
    pipe.dit = dit
    torch_gc()
    print(f"    dit ready in {(time.time()-t0)/60:.1f} min")

    pipe.dit.to(dtype=torch.bfloat16)
    import torch.nn.functional as F

    def _patch_linear_xla_safe(_name, _m):
        if not isinstance(_m, torch.nn.Linear):
            return 0
        if getattr(_m, "_xla_mixed_precision_safe", False):
            return 0
        _m._orig_forward_xla_mixed_precision_safe = _m.forward
        _m._xla_mixed_precision_safe = True
        def _linear_forward_xla_safe(x, _m=_m):
            if torch.is_tensor(x) and x.is_floating_point() and x.device.type == "xla":
                out = F.linear(x, _m.weight, None)
                if _m.bias is not None:
                    out = out + _m.bias.to(dtype=out.dtype)
                if x.dtype != torch.float32 and out.dtype != x.dtype:
                    out = out.to(dtype=x.dtype)
                return out
            return _m._orig_forward_xla_mixed_precision_safe(x)
        _m.forward = _linear_forward_xla_safe
        return 1

    _cnt = 0
    for _name, _m in pipe.dit.named_modules():
        _cnt += _patch_linear_xla_safe(_name, _m)
    print(f"[dtype-fix] patched Linear modules: {_cnt}", flush=True)

    def _to_bf16_tree_for_dit(x):
        if torch.is_tensor(x):
            if args.runtime_scalar_fix and x.device.type != "xla":
                x = x.to(GS.xla_dev)
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
        def _dit_forward_cast_inputs(*fa, **fkw):
            fa = _to_bf16_tree_for_dit(fa)
            fkw = _to_bf16_tree_for_dit(fkw)
            return pipe.dit._orig_forward_xla_dtype_patch(*fa, **fkw)
        pipe.dit.forward = _dit_forward_cast_inputs

    if args.sync_every_n_steps == 1:
        _orig_dit_forward = pipe.dit.forward
        def _dit_forward_precut(*fa, **fkw):
            torch_xla.sync()
            return _orig_dit_forward(*fa, **fkw)
        pipe.dit.forward = _dit_forward_precut
        print("[graph-cut] sync before every DiT forward", flush=True)

    def _after_step(i):
        do_sync = ((i + 1) % max(1, args.sync_every_n_steps) == 0) or (i + 1 == 8)
        if do_sync:
            torch_xla.sync()
            xm.wait_device_ops()
        now = time.time()
        if GS.step_state["t"] is not None:
            tag = "sync" if do_sync else "lazy"
            elapsed = now - GS.step_state["t"]
            print(f"    [step {i} {tag}] {elapsed:.1f}s", flush=True)
            if GS.progress_cb:
                GS.progress_cb(f"denoise step {i}: {elapsed:.1f}s")
        GS.step_state["t"] = now
    pipe._after_denoise_step = _after_step

    _orig_cache_clean = pipe._cache_clean_latents
    def _cache_clean_latents_tagged(*ca, **ckw):
        torch_xla.sync()
        t0 = time.time()
        out = _orig_cache_clean(*ca, **ckw)
        torch_xla.sync()
        xm.wait_device_ops()
        print(f"    [avc kv-build] {time.time() - t0:.1f}s", flush=True)
        if GS.progress_cb:
            GS.progress_cb(f"avc kv-build: {time.time() - t0:.1f}s")
        return out
    pipe._cache_clean_latents = _cache_clean_latents_tagged

    GS.pipe = pipe
    GS.load_stage = "Ready"
    GS.ready = True
    print("[init] TPU models loaded and ready", flush=True)
    print("[warmup] first-request warmup will run when you click Generate", flush=True)


# ---------------------------------------------------------------------------
# GPU: model init (2× T4, INT8 sharded DiT)
# ---------------------------------------------------------------------------
def init_models_gpu(args):
    """Load all models once for 2× T4 GPU. Takes ~3-5 min."""
    GS.args = args
    GS.device_type = 'gpu'
    GS.checkpoint_dir = args.checkpoint_dir
    GS.base_dir = args.base_dir or os.path.join(args.checkpoint_dir, '..', 'LongCat-Video')
    GS.save_fps = args.save_fps

    checkpoint_dir = GS.checkpoint_dir
    base_dir = GS.base_dir
    model_type = GS.model_type

    assert torch.cuda.device_count() >= 2, (
        f"GPU mode needs 2 GPUs, found {torch.cuda.device_count()}. "
        "On Kaggle pick the 'GPU T4 x2' accelerator."
    )
    dev0, dev1 = "cuda:0", "cuda:1"
    GS.dev0 = dev0
    GS.dev1 = dev1

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print("[stage 0] tokenizer / scheduler / VAE", flush=True)
    from transformers import AutoTokenizer
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan

    tokenizer = AutoTokenizer.from_pretrained(base_dir, subfolder="tokenizer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler")
    vae = AutoencoderKLWan.from_pretrained(
        base_dir, subfolder="vae", torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(dev0).eval()

    GS.tokenizer = tokenizer
    GS.scheduler = scheduler
    GS.vae = vae

    # Audio encoder (loaded once, kept for per-request use)
    print("[stage 1] loading Whisper-large-v3 audio encoder (fp16, GPU0)", flush=True)
    from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
    from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline

    audio_model_path = os.path.join(checkpoint_dir, 'whisper-large-v3')
    audio_encoder = get_audio_encoder(audio_model_path, model_type).half().to(dev0)
    audio_feature_extractor = get_audio_feature_extractor(audio_model_path, model_type)

    pipe = LongCatVideoAvatarPipeline(
        tokenizer=tokenizer, text_encoder=None, vae=vae, scheduler=scheduler,
        dit=None, audio_encoder=audio_encoder,
        audio_feature_extractor=audio_feature_extractor, model_type=model_type,
    )
    pipe.device = dev0

    # Text encoder (loaded once, kept for per-request use)
    print("[stage 2] loading umT5-xxl text encoder (fp16, GPU0)", flush=True)
    from transformers import UMT5EncoderModel
    text_encoder = UMT5EncoderModel.from_pretrained(
        base_dir, subfolder="text_encoder", torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(dev0).eval()
    pipe.text_encoder = text_encoder

    # INT8 DiT sharded across both GPUs
    print("[stage 3] loading INT8 DiT sharded across cuda:0 / cuda:1", flush=True)
    t0 = time.time()
    from longcat_video.modules.quantization import load_quantized_dit_sharded
    dit = load_quantized_dit_sharded(
        checkpoint_dir, subfolder="base_model_int8",
        devices=(dev0, dev1), split_index=args.split_index,
        compute_dtype=torch.float16, cp_split_hw=[1, 1],
    )
    distill_ckpt = os.path.join(checkpoint_dir, 'lora', 'dmd_lora.safetensors')
    assert os.path.exists(distill_ckpt), f"missing DMD LoRA: {distill_ckpt}"
    dit.load_lora(distill_ckpt, "dmd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
    dit.enable_loras(["dmd"])
    pipe.dit = dit
    torch_gc()
    print(f"    dit ready in {(time.time()-t0)/60:.1f} min")

    # after-step hook (no XLA sync needed, just timing)
    def _after_step(i):
        now = time.time()
        if GS.step_state["t"] is not None:
            elapsed = now - GS.step_state["t"]
            print(f"    [step {i}] {elapsed:.1f}s", flush=True)
            if GS.progress_cb:
                GS.progress_cb(f"denoise step {i}: {elapsed:.1f}s")
        GS.step_state["t"] = now
    pipe._after_denoise_step = _after_step

    GS.pipe = pipe
    GS.load_stage = "Ready"
    GS.ready = True
    print("[init] GPU models loaded and ready", flush=True)


def init_vocal_separator():
    """Lazily load vocal separator (heavy, only needed when audio is uploaded)."""
    if GS.vocal_separator is not None:
        return GS.vocal_separator
    from audio_separator.separator import Separator
    vocal_separator_path = os.path.join(GS.checkpoint_dir, 'vocal_separator', 'Kim_Vocal_2.onnx')
    sep = Separator(
        output_dir=Path("./audio_temp_file/vocals"),
        output_single_stem="vocals",
        model_file_dir=os.path.dirname(vocal_separator_path),
    )
    sep.load_model(os.path.basename(vocal_separator_path))
    GS.vocal_separator = sep
    return sep


def extract_vocal(source_path, vocal_separator, tmp_dir):
    outputs = vocal_separator.separate(source_path)
    if len(outputs) <= 0:
        print("Audio separation failed. Using raw audio.")
        return source_path
    default_vocal_path = (Path(tmp_dir) / "vocals" / outputs[0]).resolve().as_posix()
    target_path = f"/tmp/temp_speech_{generate_random_uid()}_vocal.wav"
    shutil.move(default_vocal_path, target_path)
    return target_path


# ---------------------------------------------------------------------------
# Generation (called per Gradio request)
# ---------------------------------------------------------------------------
def generate(
    image: PIL.Image.Image,
    audio_path: str,
    prompt: str,
    resolution: str,
    num_segments: str,
    seed: int,
    progress=None,
):
    """
    Run the full avatar generation pipeline.
    Returns (video_path, log_text, segment_paths).
    """
    if not GS.ready and not GS.dummy:
        yield None, "Models not loaded yet. Wait for initialization to complete.", []
        return

    if GS.dummy:
        yield from _dummy_generate(image, audio_path, prompt, resolution, num_segments, seed, progress)
        return

    import librosa
    from diffusers.utils import load_image

    pipe = GS.pipe
    args = GS.args
    checkpoint_dir = GS.checkpoint_dir
    model_type = GS.model_type
    num_frames = GS.num_frames
    num_cond_frames = GS.num_cond_frames
    save_fps = GS.save_fps
    audio_stride = GS.audio_stride

    is_tpu = GS.device_type == 'tpu'
    if is_tpu:
        import torch_xla
        import torch_xla.core.xla_model as xm
        primary_dev = GS.xla_dev
        gen_device = "cpu"  # TPU: noise sampled on host, transferred
        audio_emb_device = primary_dev
        audio_emb_dtype = torch.bfloat16
        te_dtype = torch.bfloat16
        te_device = torch.device("cpu")
    else:
        primary_dev = GS.dev0
        gen_device = primary_dev  # GPU: generator on cuda:0
        audio_emb_device = primary_dev
        audio_emb_dtype = torch.float16
        te_dtype = torch.float16
        te_device = primary_dev

    log_lines = []
    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    # Set progress callback — also appends to log_lines for live UI
    def _progress(msg):
        log(f"    {msg}")
        if progress:
            progress(msg)
    GS.progress_cb = _progress
    GS.step_state["t"] = None

    # --- Resolution ---
    if resolution == '480p':
        height, width = 480, 832
    else:
        height, width = 768, 1280

    # --- Segments ---
    num_segments_auto = num_segments.lower() == 'auto'
    n_seg = 1 if num_segments_auto else max(1, int(num_segments))

    # --- Save uploaded files ---
    output_dir = f"/kaggle/working/outputs_gradio_{generate_random_uid()}"
    os.makedirs(output_dir, exist_ok=True)

    # Save image
    if image is None:
        yield None, "Please upload a reference image.", []
        return
    img_path = os.path.join(output_dir, "input_image.png")
    image.save(img_path)

    # --- Stage 1: vocal separation + audio embedding ---
    log("[stage 1] vocal separation + audio embedding")
    yield None, "\n".join(log_lines), []
    if progress:
        progress("Stage 1: Vocal separation + audio embedding...")

    audio_tmp_dir = Path("./audio_temp_file")
    (audio_tmp_dir / "vocals").mkdir(parents=True, exist_ok=True)

    vocal_sep = init_vocal_separator()
    temp_vocal_path = extract_vocal(audio_path, vocal_sep, audio_tmp_dir)
    assert temp_vocal_path is not None and os.path.exists(temp_vocal_path), "No vocal detected"

    speech_array, sr = librosa.load(temp_vocal_path, sr=16000)
    source_duration = len(speech_array) / sr

    if num_segments_auto:
        if source_duration * save_fps <= num_frames:
            n_seg = 1
        else:
            n_seg = max(1, math.ceil(
                1 + (source_duration * save_fps - num_frames) / (num_frames - num_cond_frames)))
        log(f"    [auto] audio {source_duration:.1f}s -> {n_seg} segment(s)")

    generate_duration = num_frames / save_fps + (n_seg - 1) * (num_frames - num_cond_frames) / save_fps
    added = math.ceil((generate_duration - source_duration) * sr)
    if added > 0:
        speech_array = np.append(speech_array, [0.] * added)
    log(f"    audio {source_duration:.1f}s, target video {generate_duration:.1f}s "
        f"({n_seg} segment(s), {num_frames + (n_seg-1)*(num_frames-num_cond_frames)} frames)")

    audio_emb_compute_device = "cpu" if is_tpu else GS.dev0
    with torch.no_grad():
        full_audio_emb = pipe.get_audio_embedding(
            speech_array, fps=save_fps * audio_stride, device=audio_emb_compute_device,
            sample_rate=sr, model_type=model_type,
        )
    if torch.isnan(full_audio_emb).any():
        yield None, "Broken audio embedding with NaN values.", []
        return
    full_audio_emb = full_audio_emb.float().cpu()
    log(f"    audio embedding: {tuple(full_audio_emb.shape)}")

    if os.path.exists(temp_vocal_path):
        os.remove(temp_vocal_path)
    torch_gc()

    # --- Stage 2: text encoding ---
    yield None, "\n".join(log_lines), []
    if progress:
        progress("Stage 2: Text encoding...")
    log("[stage 2] text encoding")

    negative_prompt = GS.negative_prompt
    with torch.no_grad():
        pe, pm, npe, npm = pipe.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1, max_sequence_length=512,
            dtype=te_dtype, device=te_device,
        )
    pipe.set_cached_text_embeddings(pe.cpu(), pm.cpu(), npe.cpu(), npm.cpu())
    torch_gc()

    # --- Audio slicing helper ---
    indices = torch.arange(2 * 2 + 1) - 2
    def slice_audio_emb(start, end):
        ci = torch.arange(start, end, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
        ci = torch.clamp(ci, min=0, max=full_audio_emb.shape[0] - 1)
        return full_audio_emb[ci][None, ...].to(device=audio_emb_device, dtype=audio_emb_dtype)

    # --- Generator ---
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)

    use_distill = True
    text_guidance_scale = 1.0
    audio_guidance_scale = 1.0
    num_inference_steps = args.num_inference_steps

    # --- Warmup (TPU only, first request) ---
    if is_tpu and args.warmup and not getattr(GS, '_warmed_up', False):
        if progress:
            progress("Warmup: compiling XLA graphs (one-time, ~5 min)...")
        log("[warmup] compiling graphs on throwaway data...")
        t_warm = time.time()
        warmup_generator = torch.Generator(device="cpu").manual_seed(0)
        warmup_audio_emb = slice_audio_emb(0, audio_stride * num_frames)
        common_warm = dict(
            prompt=prompt, negative_prompt=negative_prompt,
            num_frames=num_frames, num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale, audio_guidance_scale=audio_guidance_scale,
            generator=warmup_generator, output_type='both', audio_emb=warmup_audio_emb, use_distill=use_distill,
        )
        with torch.no_grad():
            warm_image = load_image(img_path)
            warm_output, warm_latent = pipe.generate_ai2v(
                image=warm_image, resolution=resolution, **common_warm)
        warm_video = [PIL.Image.fromarray((warm_output[0][i] * 255).astype(np.uint8)) for i in range(warm_output.shape[1])]
        del warm_output
        torch_gc()

        if n_seg > 1:
            # Warmup AVC with the real ai2v output dims (bucket resolution),
            # not the default height/width: otherwise the actual AVC run hits
            # a different graph shape -> recompilation + a second resident
            # compiled program eating HBM.
            warm_width, warm_height = warm_video[0].size
            warm_ref_latent = warm_latent[:, :, :1].clone() if warm_latent is not None else None
            with torch.no_grad():
                pipe.generate_avc(
                    video=warm_video, video_latent=warm_latent if warm_latent is not None else torch.zeros(1,16,1,warm_height//8,warm_width//8).to(primary_dev).to(torch.bfloat16),
                    prompt=prompt, negative_prompt=negative_prompt,
                    height=warm_height, width=warm_width,
                    num_frames=num_frames, num_cond_frames=num_cond_frames,
                    num_inference_steps=num_inference_steps,
                    text_guidance_scale=text_guidance_scale, audio_guidance_scale=audio_guidance_scale,
                    generator=warmup_generator, output_type='latent',
                    use_kv_cache=True, offload_kv_cache=args.offload_kv_cache,
                    enhance_hf=False,
                    audio_emb=warmup_audio_emb, ref_latent=warm_ref_latent,
                    ref_img_index=args.ref_img_index, mask_frame_range=args.mask_frame_range,
                    use_distill=use_distill,
                )
            del warm_ref_latent
        del warm_latent, warm_video
        pipe._clear_cache()
        torch_gc()
        torch_xla.sync()
        xm.wait_device_ops()
        log(f"[warmup] done in {(time.time()-t_warm)/60:.1f} min")
        GS._warmed_up = True

    # --- Stage 4: generation ---
    audio_start_idx = 0
    audio_end_idx = audio_start_idx + audio_stride * num_frames
    segment_paths = []

    # Segment 1: ai2v
    log(f"[stage 4] segment 1/{n_seg} (ai2v) ...")
    yield None, "\n".join(log_lines), []
    if progress:
        progress(f"Stage 4: Segment 1/{n_seg} (ai2v) — denoising + VAE decode...")
    t0 = time.time()
    GS.step_state["t"] = time.time()

    import threading as _threading, queue as _queue
    audio_emb = slice_audio_emb(audio_start_idx, audio_end_idx)
    common = dict(
        prompt=prompt, negative_prompt=negative_prompt,
        num_frames=num_frames, num_inference_steps=num_inference_steps,
        text_guidance_scale=text_guidance_scale, audio_guidance_scale=audio_guidance_scale,
        generator=generator, output_type='both', audio_emb=audio_emb, use_distill=use_distill,
    )
    _ai2v_q = _queue.Queue()
    def _run_ai2v():
        try:
            with torch.no_grad():
                _img = load_image(img_path)
                result = pipe.generate_ai2v(image=_img, resolution=resolution, **common)
            _ai2v_q.put(("ok", result))
        except Exception as _e:
            import traceback as _tb
            _ai2v_q.put(("err", _tb.format_exc()))
    _ai2v_t = _threading.Thread(target=_run_ai2v, daemon=True)
    _ai2v_t.start()
    while _ai2v_t.is_alive():
        import time as _t; _t.sleep(3)
        yield None, "\n".join(log_lines), []
    _ai2v_status, _ai2v_result = _ai2v_q.get()
    if _ai2v_status == "err":
        yield None, "\n".join(log_lines) + "\nERROR in ai2v:\n" + _ai2v_result, []
        return
    output, latent = _ai2v_result

    output = output[0]
    video = [PIL.Image.fromarray((output[i] * 255).astype(np.uint8)) for i in range(output.shape[0])]
    del output
    torch_gc()

    from longcat_video.audio_process.torch_utils import save_video_ffmpeg
    seg1_path = os.path.join(output_dir, "segment_1")
    save_video_ffmpeg(
        torch.from_numpy(np.array(video)),
        seg1_path, audio_path, fps=save_fps, quality=5,
    )
    segment_paths.append(seg1_path + ".mp4")
    log(f"    segment 1 done in {(time.time()-t0)/60:.1f} min")
    yield None, "\n".join(log_lines), list(segment_paths)

    # Save first-frame preview
    preview_path = os.path.join(output_dir, "segment_1_frame0.png")
    video[0].save(preview_path)

    all_generated_frames = list(video)
    current_video = video
    ref_latent = latent[:, :, :1].clone()
    width, height = video[0].size

    # Free segment-1 buffers
    pipe._clear_cache()
    torch_gc()
    if is_tpu:
        torch_xla.sync()
        xm.wait_device_ops()

    # Segments 2+: avc
    for segment_idx in range(1, n_seg):
        if progress:
            progress(f"Stage 4: Segment {segment_idx+1}/{n_seg} (avc) — denoising + VAE decode...")
        log(f"[stage 4] segment {segment_idx+1}/{n_seg} (avc) ...")
        yield None, "\n".join(log_lines), list(segment_paths)
        t0 = time.time()
        GS.step_state["t"] = time.time()

        audio_start_idx = audio_start_idx + audio_stride * (num_frames - num_cond_frames)
        audio_end_idx = audio_start_idx + audio_stride * num_frames
        audio_emb = slice_audio_emb(audio_start_idx, audio_end_idx)

        _avc_q = _queue.Queue()
        _cur_video = current_video; _cur_latent = latent; _cur_ref = ref_latent
        def _run_avc():
            try:
                with torch.no_grad():
                    result = pipe.generate_avc(
                        video=_cur_video, video_latent=_cur_latent,
                        prompt=prompt, negative_prompt=negative_prompt,
                        height=height, width=width,
                        num_frames=num_frames, num_cond_frames=num_cond_frames,
                        num_inference_steps=num_inference_steps,
                        text_guidance_scale=text_guidance_scale, audio_guidance_scale=audio_guidance_scale,
                        generator=generator, output_type='both',
                        use_kv_cache=True, offload_kv_cache=args.offload_kv_cache,
                        enhance_hf=False,
                        audio_emb=audio_emb, ref_latent=_cur_ref,
                        ref_img_index=args.ref_img_index, mask_frame_range=args.mask_frame_range,
                        use_distill=use_distill,
                    )
                _avc_q.put(("ok", result))
            except Exception as _e:
                import traceback as _tb
                _avc_q.put(("err", _tb.format_exc()))
        _avc_t = _threading.Thread(target=_run_avc, daemon=True)
        _avc_t.start()
        while _avc_t.is_alive():
            import time as _t; _t.sleep(3)
            yield None, "\n".join(log_lines), list(segment_paths)
        _avc_status, _avc_result = _avc_q.get()
        if _avc_status == "err":
            yield None, "\n".join(log_lines) + "\nERROR in avc:\n" + _avc_result, list(segment_paths)
            return
        output, latent = _avc_result

        output = output[0]
        new_video = [PIL.Image.fromarray((output[i] * 255).astype(np.uint8)) for i in range(output.shape[0])]
        del output

        all_generated_frames.extend(new_video[num_cond_frames:])
        current_video = new_video

        pipe._clear_cache()
        torch_gc()

        seg_path = os.path.join(output_dir, f"video_continue_{segment_idx+1}")
        save_video_ffmpeg(
            torch.from_numpy(np.array(all_generated_frames)),
            seg_path, audio_path, fps=save_fps, quality=5,
        )
        segment_paths.append(seg_path + ".mp4")
        log(f"    segment {segment_idx+1} done in {(time.time()-t0)/60:.1f} min "
            f"(total {len(all_generated_frames)} frames = {len(all_generated_frames)/save_fps:.1f}s)")
        yield None, "\n".join(log_lines), list(segment_paths)

    # Final full video
    final_path = os.path.join(output_dir, "final_video")
    save_video_ffmpeg(
        torch.from_numpy(np.array(all_generated_frames)),
        final_path, audio_path, fps=save_fps, quality=5,
    )
    final_video = final_path + ".mp4"
    log(f"[done] final video: {final_video} ({len(all_generated_frames)} frames)")

    GS.progress_cb = None
    yield final_video, "\n".join(log_lines), segment_paths


def _dummy_generate(image, audio_path, prompt, resolution, num_segments, seed, progress):
    """Fake generation for UI testing without TPU."""
    log_lines = []
    def log(msg):
        log_lines.append(msg)

    if progress:
        progress("Dummy mode: simulating generation...")

    for stage in ["vocal separation", "text encoding", "denoise step 0", "denoise step 1",
                  "vae decode", "saving video"]:
        log(f"[dummy] {stage}...")
        time.sleep(0.5)
        if progress:
            progress(f"Dummy: {stage}")

    # Create a dummy video (just repeat the input image)
    output_dir = f"/tmp/dummy_output_{generate_random_uid()}"
    os.makedirs(output_dir, exist_ok=True)

    if image is None:
        image = PIL.Image.new('RGB', (832, 480), color='blue')
    frames = [np.array(image.resize((832, 480)))] * 25
    dummy_video_path = os.path.join(output_dir, "dummy.mp4")

    try:
        import imageio
        writer = imageio.get_writer(dummy_video_path, fps=25, codec='libx264')
        for f in frames:
            writer.append_data(f)
        writer.close()
    except Exception:
        yield None, "\n".join(log_lines) + "\n[dummy] could not save video (imageio not available)", []
        return

    log(f"[dummy] saved to {dummy_video_path}")
    yield dummy_video_path, "\n".join(log_lines), [dummy_video_path]


# ---------------------------------------------------------------------------
# Session persistence — per-session-ID dirs, survive tunnel drops / reloads
# ---------------------------------------------------------------------------
SESSION_ROOT = ("/kaggle/working/gradio_sessions"
                if os.path.isdir("/kaggle/working") else "/tmp/gradio_sessions")


def _new_session_id():
    return time.strftime("%m%d-%H%M%S") + "-" + str(random.randint(100, 999))


def _session_dir(sid):
    d = os.path.join(SESSION_ROOT, sid)
    os.makedirs(d, exist_ok=True)
    return d


def _session_path(sid, name):
    return os.path.join(_session_dir(sid), name)


def _latest_session_id():
    if not os.path.isdir(SESSION_ROOT):
        return None
    dirs = [p for p in Path(SESSION_ROOT).iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime).name


def _session_save_json(sid, **kwargs):
    """Merge kwargs into <session>/session.json."""
    path = _session_path(sid, "session.json")
    data = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            data = {}
    data.update(kwargs)
    with open(path, "w") as f:
        json.dump(data, f)


def _session_load_json(sid):
    path = _session_path(sid, "session.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _session_save_image(sid, img):
    if not sid:
        return
    try:
        p = _session_path(sid, "image.png")
        if img is None:
            if os.path.exists(p):
                os.remove(p)
        else:
            img.save(p)
    except Exception as e:
        print(f"[session] image save failed: {e}", flush=True)


def _session_save_audio(sid, audio_path):
    if not sid:
        return
    try:
        # remove any previous audio.* file
        for old in Path(_session_dir(sid)).glob("audio.*"):
            old.unlink(missing_ok=True)
        if audio_path and os.path.exists(audio_path):
            p = _session_path(sid, "audio" + (os.path.splitext(audio_path)[1] or ".wav"))
            shutil.copy(audio_path, p)
            _session_save_json(sid, audio=p)
        else:
            _session_save_json(sid, audio=None)
    except Exception as e:
        print(f"[session] audio save failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# Background jobs — generation keeps running even if the client disconnects;
# re-attach any time with the session ID.
# ---------------------------------------------------------------------------
JOBS = {}  # sid -> {"status": "running"|"done"|"error", "video", "log", "segments"}
_GEN_LOCK = threading.Lock()  # single accelerator: one generation at a time


def _run_job(sid, image, audio_path, prompt, resolution, num_segments, seed):
    job = JOBS[sid]
    if _GEN_LOCK.locked():
        job["log"] = "[queue] another job is running, waiting for it to finish..."
    with _GEN_LOCK:
        try:
            for video, log_text, segs in generate(
                    image, audio_path, prompt, resolution, num_segments, seed):
                job["video"], job["log"], job["segments"] = video, log_text, list(segs)
                _session_save_json(sid, job_video=video, job_log=log_text,
                                   job_segments=list(segs), job_status="running")
            job["status"] = "done"
            _session_save_json(sid, job_status="done", job_video=job["video"],
                               job_log=job["log"], job_segments=job["segments"])
        except Exception:
            import traceback as _tb
            job["log"] = (job.get("log") or "") + "\nERROR:\n" + _tb.format_exc()
            job["status"] = "error"
            _session_save_json(sid, job_status="error", job_log=job["log"])


def _start_or_attach(sid, image, audio_path, prompt, resolution, num_segments, seed):
    """Start a background generation job for this session (or attach to a
    running one) and stream its progress. Safe to disconnect and re-attach."""
    sid = (sid or "").strip() or _new_session_id()
    job = JOBS.get(sid)
    if job is None or job["status"] != "running":
        job = {"status": "running", "video": None, "log": "", "segments": []}
        JOBS[sid] = job
        threading.Thread(target=_run_job, daemon=True, name=f"job-{sid}",
                         args=(sid, image, audio_path, prompt,
                               resolution, num_segments, seed)).start()
    while job["status"] == "running":
        yield job["video"], job["log"], list(job["segments"])
        time.sleep(2)
    yield job["video"], job["log"], list(job["segments"])


def _restore_session(sid):
    """Restore inputs + results for a session ID; if its job is still running,
    keep streaming progress (re-attach). Yields 9 component updates."""
    import gradio as gr
    sid = (sid or "").strip()
    noop = gr.update()
    if not sid or not os.path.isdir(os.path.join(SESSION_ROOT, sid)):
        yield (noop,) * 8 + (gr.update(value=f"[session] '{sid}' not found"),)
        return

    data = _session_load_json(sid)
    img_p = _session_path(sid, "image.png")
    image = img_p if os.path.exists(img_p) else None
    audio = data.get("audio")
    if audio and not os.path.exists(audio):
        audio = None
    inputs = (
        gr.update(value=image) if image else noop,
        gr.update(value=audio) if audio else noop,
        gr.update(value=data["prompt"]) if data.get("prompt") else noop,
        gr.update(value=data["resolution"]) if data.get("resolution") else noop,
        gr.update(value=data["segments"]) if data.get("segments") else noop,
        gr.update(value=data["seed"]) if data.get("seed") is not None else noop,
    )
    no_inputs = (noop,) * 6

    job = JOBS.get(sid)
    if job and job["status"] == "running":
        first = True
        while job["status"] == "running":
            yield (inputs if first else no_inputs) + (
                gr.update(value=job["video"]) if job["video"] else noop,
                gr.update(value=list(job["segments"])),
                gr.update(value=job["log"]),
            )
            first = False
            time.sleep(2)
        yield no_inputs + (
            gr.update(value=job["video"]) if job["video"] else noop,
            gr.update(value=list(job["segments"])),
            gr.update(value=job["log"]),
        )
        return

    # No live job — restore last persisted results
    video = data.get("job_video") or data.get("last_video")
    if video and not os.path.exists(video):
        video = None
    segs = [s for s in (data.get("job_segments") or []) if os.path.exists(s)]
    log_text = data.get("job_log") or f"[session] restored '{sid}'"
    yield inputs + (
        gr.update(value=video) if video else noop,
        gr.update(value=segs),
        gr.update(value=log_text),
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui():
    import gradio as gr

    with gr.Blocks(title="LongCat Video Avatar", css="@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.6; } }") as app:
        gr.Markdown("""
        # LongCat Video Avatar

        Upload a reference image and audio to generate a lip-synced avatar video.
        Models load once at startup; subsequent generations are faster.
        Supports **TPU v5e-8** and **2× T4 GPU** backends.
        """)

        # Status banner at top
        status_html = gr.HTML(
            value='<div style="padding:10px 16px; border-radius:8px; background:#f0f0f0; color:#666; font-weight:bold; font-size:15px; text-align:center; border:2px solid #ddd;">Initializing...</div>'
        )

        # Session bar — copy the ID; paste it back after a tunnel/URL change
        # to restore uploads, settings and any still-running generation.
        with gr.Row():
            sid_box = gr.Textbox(
                label="Session ID (保存好，换链接后粘回来可找回上传/进度/结果)",
                value="", max_lines=1, scale=3,
            )
            restore_btn = gr.Button("恢复 Session", scale=1)
            new_btn = gr.Button("新 Session", scale=1)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Reference Image", type="pil", height=300)
                input_audio = gr.Audio(label="Speech Audio", type="filepath")
                prompt_text = gr.Textbox(
                    label="Prompt",
                    value="A western man stands on stage under dramatic lighting, holding a microphone close to their mouth. Wearing a vibrant red jacket with gold embroidery, the singer is speaking while smoke swirls around them, creating a dynamic and atmospheric scene.",
                    lines=3,
                )
                with gr.Row():
                    res_dd = gr.Dropdown(["480p", "720p"], value="480p", label="Resolution")
                    seg_dd = gr.Dropdown(["auto", "1", "2", "3", "4", "5", "6"], value="auto", label="Segments")
                seed_slider = gr.Slider(0, 999999, value=42, step=1, label="Seed")
                generate_btn = gr.Button("Generate Video", variant="primary", size="lg", interactive=False)

            with gr.Column(scale=1):
                output_video = gr.Video(label="Generated Video")
                with gr.Accordion("Segment Videos", open=False):
                    segment_gallery = gr.Files(label="Per-segment videos")
                with gr.Accordion("Logs", open=True):
                    log_box = gr.Textbox(label="Log", lines=20, max_lines=50, interactive=False)

        def _check_status():
            if GS.dummy:
                return "Ready (dummy mode)"
            if GS.ready:
                dev = GS.device_type.upper() if GS.device_type else "?"
                return f"Ready — models loaded ({dev})"
            # Show detailed loading progress
            stage = getattr(GS, "load_stage", "Initializing...")
            return f"Loading models... ({stage})"

        def _status_html():
            if GS.dummy:
                return '<div style="padding:10px 16px; border-radius:8px; background:#d4edda; color:#155724; font-weight:bold; font-size:15px; text-align:center; border:2px solid #c3e6cb;">✅ Ready (dummy mode)</div>'
            if GS.ready:
                dev = GS.device_type.upper() if GS.device_type else "?"
                return f'<div style="padding:10px 16px; border-radius:8px; background:#d4edda; color:#155724; font-weight:bold; font-size:15px; text-align:center; border:2px solid #c3e6cb;">✅ Ready — models loaded ({dev})</div>'
            stage = getattr(GS, "load_stage", "Initializing...")
            return f'<div style="padding:10px 16px; border-radius:8px; background:#fff3cd; color:#856404; font-weight:bold; font-size:15px; text-align:center; border:2px solid #ffeaa7; animation:pulse 1.5s infinite;">⏳ Loading models... ({stage})</div>'

        def _update_ui():
            html = _status_html()
            btn_update = gr.update(interactive=(GS.ready or GS.dummy))
            return html, btn_update

        _beep_html = gr.HTML("", visible=False, elem_id="beep-trigger")

        # --- Session persistence: save inputs (per session ID) as they change ---
        input_image.change(fn=_session_save_image, inputs=[sid_box, input_image])
        input_audio.change(fn=_session_save_audio, inputs=[sid_box, input_audio])
        prompt_text.change(fn=lambda s, v: _session_save_json(s, prompt=v) if s else None,
                           inputs=[sid_box, prompt_text])
        res_dd.change(fn=lambda s, v: _session_save_json(s, resolution=v) if s else None,
                      inputs=[sid_box, res_dd])
        seg_dd.change(fn=lambda s, v: _session_save_json(s, segments=v) if s else None,
                      inputs=[sid_box, seg_dd])
        seed_slider.change(fn=lambda s, v: _session_save_json(s, seed=v) if s else None,
                           inputs=[sid_box, seed_slider])

        _restore_outputs = [input_image, input_audio, prompt_text, res_dd, seg_dd,
                            seed_slider, output_video, segment_gallery, log_box]

        # On (re)load: pick up the most recent session (or create one), then
        # restore it — including re-attaching to a still-running generation.
        def _on_load():
            sid = _latest_session_id() or _new_session_id()
            _session_dir(sid)
            print(f"[session] active session: {sid}", flush=True)
            return sid

        app.load(fn=_on_load, outputs=sid_box).then(
            fn=_restore_session, inputs=sid_box, outputs=_restore_outputs,
        )

        restore_btn.click(fn=_restore_session, inputs=sid_box, outputs=_restore_outputs)
        new_btn.click(fn=lambda: _new_session_id(), outputs=sid_box)

        generate_btn.click(
            fn=_start_or_attach,
            inputs=[sid_box, input_image, input_audio, prompt_text, res_dd, seg_dd, seed_slider],
            outputs=[output_video, log_box, segment_gallery],
        ).then(
            fn=lambda: '<script>new Audio("data:audio/wav;base64,UklGRl9vT19XQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=").play()</script>',
            outputs=_beep_html,
        )

        # Periodic status update via Timer (Gradio 6.0+)
        timer = gr.Timer(value=5)
        timer.tick(fn=_update_ui, outputs=[status_html, generate_btn])

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _init_backend(args):
    """Start model loading in a background thread; GS is updated in place."""
    if args.dummy:
        print("[dummy] running without accelerator — UI only")
        GS.dummy = True
        GS.ready = True
        return

    dev_type = detect_device(args)
    GS.device_type = dev_type
    print(f"[main] detected device: {dev_type}", flush=True)

    if dev_type == 'tpu':
        def _init_bg():
            try:
                init_environment_tpu(args)
                init_models_tpu(args)
            except Exception as e:
                print(f"[init] FATAL: {e}", flush=True)
                import traceback
                traceback.print_exc()
    else:
        def _init_bg():
            try:
                init_models_gpu(args)
            except Exception as e:
                print(f"[init] FATAL: {e}", flush=True)
                import traceback
                traceback.print_exc()

    t = threading.Thread(target=_init_bg, daemon=True)
    t.start()


def run_gradio_app(
    checkpoint_dir: str = "/dev/shm/weights/LongCat-Video-Avatar-1.5",
    base_dir: str = None,
    device: str = "auto",
    share: bool = False,
    inline: bool = True,
    warmup: bool = False,
    num_inference_steps: int = 8,
    offload_kv_cache: bool = True,
    dummy: bool = False,
):
    """Notebook-friendly entry point.

    Example:
        from gradio_server import run_gradio_app
        run_gradio_app(checkpoint_dir="/dev/shm/weights/LongCat-Video-Avatar-1.5",
                       base_dir="/dev/shm/weights/LongCat-Video")
    """
    args = _parse_args([])  # avoid parsing Jupyter kernel arguments
    args.device = device
    args.checkpoint_dir = checkpoint_dir
    args.base_dir = base_dir
    args.share = share
    args.warmup = warmup
    args.num_inference_steps = num_inference_steps
    args.offload_kv_cache = offload_kv_cache
    args.dummy = dummy

    _init_backend(args)

    app = build_ui()
    import gradio as gr
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        inline=inline,
        allowed_paths=["/kaggle/working", "/tmp"],
        theme=gr.themes.Soft() if hasattr(gr, 'themes') else None,
    )


def main():
    args = _parse_args()
    _init_backend(args)

    app = build_ui()
    import gradio as gr
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        allowed_paths=["/kaggle/working", "/tmp"],
        theme=gr.themes.Soft() if hasattr(gr, 'themes') else None,
    )


if __name__ == "__main__":
    main()
