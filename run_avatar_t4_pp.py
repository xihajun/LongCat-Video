"""
LongCat-Video-Avatar 1.5 inference for Kaggle 2x T4 (15GB) + 30GB CPU RAM.

Key differences vs. the official run_demo_avatar_single_audio_to_video.py:

1.  SINGLE PROCESS, TWO GPUs (pipeline / layer parallelism).
    The official multi-GPU path is *context parallelism*: it splits the token
    sequence but replicates the FULL 13.6B DiT on every GPU, so it can never
    fit a T4. Here we instead split the 48 transformer blocks across the two
    GPUs (blocks[0:split] -> cuda:0, blocks[split:] + final_layer -> cuda:1)
    and move activations between them inside wrapped forward() calls.

2.  LOW-RAM INT8 LOADER.
    The official `load_quantized_dit` first materializes the whole model in
    fp32 on CPU (~54 GB!) and then loads all 15.9 GB of int8 shards into one
    CPU dict -> instant SIGKILL on 30 GB Kaggle RAM. We instead build the
    model on the *meta* device (0 bytes), then stream each safetensors shard
    and assign tensors directly onto their target GPU. CPU peak ~5 GB.

3.  SEQUENTIAL ENCODER LIFECYCLE.
    Whisper-large-v3 (3.1 GB) and UMT5-XXL (11.4 GB fp16) are loaded one at a
    time on cuda:0, used once (audio embedding / prompt embedding), then
    freed BEFORE the DiT is loaded. Prompt embeddings are cached and
    `pipe.encode_prompt` is monkey-patched so later segments never touch the
    (now deleted) text encoder.

4.  T4 (SM75) COMPATIBILITY.
    - flash-attn requires Ampere+; this repo has NO SDPA fallback, so we
      force `enable_xformers=True` (cutlass kernels support SM75).
    - bf16 has no tensor cores on T4 -> everything is cast to fp16.
      (LayerNorm / modulation already run in fp32 inside the model.)

5.  LONG-VIDEO MEMORY HYGIENE.
    - KV cache for video continuation is offloaded to CPU by default.
    - Each segment is written to its own mp4; the final video is produced by
      a lossless ffmpeg concat + audio mux, so CPU RAM does not grow with
      video length (the official demo re-materializes ALL frames as one
      giant numpy tensor every segment).

Run from the LongCat-Video repo root, e.g.:

    python run_avatar_t4_pp.py \
        --checkpoint_dir /kaggle/temp/weights/LongCat-Video-Avatar-1.5 \
        --stage_1 ai2v \
        --input_json assets/avatar/single_example_1.json \
        --output_dir /kaggle/working/outputs_avatar \
        --num_segments auto
"""

import os
import sys
import gc
import json
import math
import time
import types
import random
import argparse
import subprocess
import atexit
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import PIL.Image

import torch
import torch.nn as nn
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Environment hardening (must run before heavy imports)
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers.utils import load_image
from safetensors.torch import load_file
from accelerate import init_empty_weights

from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from longcat_video.modules.quantization import QuantizedLinear, DEFAULT_SKIP_PATTERNS
from longcat_video.context_parallel import context_parallel_util

import librosa
from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
from audio_separator.separator import Separator

from fp16_range_toolkit import (
    print_block_anatomy, install_block_stats, install_soft_clamp,
    install_nonfinite_tracer,
)

DEV0 = "cuda:0"
DEV1 = "cuda:1"

NEGATIVE_PROMPT = (
    "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, "
    "ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, "
    "many people in the background, walking backwards"
)


def log(msg):
    print(f"[t4pp {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def torch_gc():
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def report_vram(tag=""):
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024 ** 3
        reserv = torch.cuda.memory_reserved(i) / 1024 ** 3
        log(f"VRAM[{tag}] cuda:{i} allocated={alloc:.2f}GB reserved={reserv:.2f}GB")


# ---------------------------------------------------------------------------
# 1. Low-RAM INT8 loader with 2-GPU block split
# ---------------------------------------------------------------------------
def _key_device(name: str, split_index: int) -> str:
    """Map a state-dict key / module name to its target device."""
    if name.startswith("blocks."):
        try:
            idx = int(name.split(".")[1])
        except (IndexError, ValueError):
            return DEV0
        return DEV0 if idx < split_index else DEV1
    if name.startswith("final_layer"):
        return DEV1
    return DEV0  # x_embedder / t_embedder / y_embedder / audio_proj / misc


def _deep_to(obj, dev):
    """Recursively move every tensor inside nested tuples/lists/dicts to dev.

    The previous version only moved top-level tensor args, so tensors nested
    inside tuples/lists (e.g. the text condition passed positionally, or
    kv_cache tuples) stayed on the wrong GPU. That is exactly what produced
    the 'attn_bias cuda:0 vs query cuda:1' crash: the xformers attention bias
    is rebuilt from seqlen tensors that must live on the query's device.
    Same-device .to() is a no-op, so this is free within a device's own blocks
    and only actually copies at the split boundary.
    """
    if torch.is_tensor(obj):
        return obj.to(dev, non_blocking=True)
    if isinstance(obj, tuple):
        return tuple(_deep_to(o, dev) for o in obj)
    if isinstance(obj, list):
        return [_deep_to(o, dev) for o in obj]
    if isinstance(obj, dict):
        return {k: _deep_to(v, dev) for k, v in obj.items()}
    return obj


def _wrap_inputs_to_device(module: nn.Module, device: str):
    """Wrap module.forward so every tensor argument (including nested ones and
    the module's own module-buffers) lives on `device` before the call."""
    orig = module.forward
    dev = torch.device(device)

    def fwd(*args, **kwargs):
        args = _deep_to(args, dev)
        kwargs = _deep_to(kwargs, dev)
        # xformers 0.0.29: BlockDiagonalMask.from_seqlens() builds its seqstart
        # tensors via _get_default_bias_device() == bare torch.device("cuda")
        # == the CURRENT device (cuda:0), while q/k/v of blocks[split:] live on
        # cuda:1 -> "Attention bias and Query/Key/Value should be on the same
        # device". The mask is built INSIDE the block and is not a tensor, so
        # _deep_to cannot fix it; running the block under its own device
        # context makes the bias land on the right GPU.
        with torch.cuda.device(dev):
            return orig(*args, **kwargs)

    module.forward = fwd


def install_chunked_ffn(model, chunk_tokens: int):
    """Run each block's FFN in sequential chunks along the token dimension.

    The FFN (w2(silu(w1(x)) * w3(x))) is pointwise over tokens, so chunking is
    mathematically exact. It is THE activation-peak killer on T4: at 480p the
    first segment has 37,440 tokens and ffn inner dim ~10.8k, so each of the
    four intermediates (w1 out, silu, w3 out, product) is a ~774 MB fp16
    tensor; chunking by ~1/4 removes ~2.3 GB from the per-block peak. The
    LoRA-wrapped w1/w2/w3 run per chunk too, shrinking their temporaries as
    well. Cost: <2% speed (weights are dequantized once per chunk).
    """
    if chunk_tokens <= 0:
        log("FFN chunking DISABLED (--ffn_chunk_tokens 0).")
        return
    n = 0
    for block in model.blocks:
        ffn = getattr(block, "ffn", None)
        if ffn is None:
            continue

        def _make(orig):
            def fwd(x, *a, **kw):
                t = x.shape[-2]
                if t <= chunk_tokens:
                    return orig(x, *a, **kw)
                return torch.cat(
                    [orig(x[..., i:i + chunk_tokens, :], *a, **kw)
                     for i in range(0, t, chunk_tokens)],
                    dim=-2,
                )
            return fwd

        ffn.forward = _make(ffn.forward)
        n += 1
    log(f"Chunked FFN installed on {n}/{len(model.blocks)} blocks "
        f"(chunk={chunk_tokens} tokens).")


def _empty_cache_all():
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()


def reset_peaks():
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def report_peaks(tag=""):
    for i in range(torch.cuda.device_count()):
        pk = torch.cuda.max_memory_allocated(i) / 1024 ** 3
        rs = torch.cuda.max_memory_reserved(i) / 1024 ** 3
        log(f"PEAK[{tag}] cuda:{i} allocated={pk:.2f}GB reserved={rs:.2f}GB")


def _pixel_report(output, tag):
    """Decoded-pixel telemetry + black-frame alarm.

    `output` is the postprocessed video array in [0,1], shape (T,H,W,C).
    A healthy talking-head segment has mean ~0.2-0.6. If frames after the
    conditioning frame are ~0, fp16 overflow killed the latents upstream
    (the sanitizer should prevent this; if you still see the alarm, send me
    the log — the 'non-finite in blocks[i]' lines pinpoint the layer).
    """
    arr = np.asarray(output)
    t = arr.shape[0]
    head = arr[: min(2, t)].mean()
    tail = arr[min(2, t):].mean() if t > 2 else head
    log(f"PIXELS[{tag}] shape={arr.shape} min={arr.min():.3f} max={arr.max():.3f} "
        f"mean(first2)={head:.3f} mean(rest)={tail:.3f}")
    if not np.isfinite(arr).all():
        log(f"    !! PIXELS[{tag}] contain NaN/Inf")
    if tail < 0.02:
        log(f"    !! BLACK-FRAME ALARM[{tag}]: generated frames are ~all zero. "
            f"fp16 overflow upstream — check 'non-finite' lines above.")


def install_dit_debug(model, level: int = 1, empty_cache: bool = True):
    """Per-DiT-forward heartbeat: call #, per-GPU alloc/peak, wall time.

    With distill (8 steps, CFG off) this prints ~8-10 lines per segment, so it
    is cheap but tells you exactly where a run is and how close to the limit
    each GPU sits. level>=2 additionally prints per-block watermarks on the
    first two forwards (blocks 0, 6, 12, ..., 47).
    """
    state = {"calls": 0}
    orig = model.forward

    def fwd(*a, **kw):
        state["calls"] += 1
        t0 = time.time()
        out = orig(*a, **kw)
        if empty_cache:
            _empty_cache_all()
        if level >= 1:
            parts = []
            for i in range(torch.cuda.device_count()):
                al = torch.cuda.memory_allocated(i) / 1024 ** 3
                pk = torch.cuda.max_memory_allocated(i) / 1024 ** 3
                parts.append(f"cuda:{i} {al:.2f}/{pk:.2f}GB")
            t_out = out[0] if isinstance(out, tuple) else out
            amax = float(t_out.abs().max()) if torch.is_tensor(t_out) else float("nan")
            fin = bool(torch.isfinite(t_out).all()) if torch.is_tensor(t_out) else True
            log(f"DiT fwd #{state['calls']:>3} done in {time.time() - t0:5.1f}s "
                f"| alloc/peak: {' | '.join(parts)} | out absmax={amax:.1f} finite={fin}")
        return out

    model.forward = fwd

    if level >= 2:
        split = getattr(model, "_t4pp_split_index", len(model.blocks))
        blk_state = {"fwd": 0}
        for i, block in enumerate(model.blocks):
            if i % 6 != 0 and i != len(model.blocks) - 1 and i != split:
                continue

            def _mk(orig, i, dev_idx):
                def bfwd(*a, **kw):
                    if i == 0:
                        blk_state["fwd"] += 1
                    if blk_state["fwd"] <= 2:
                        al = torch.cuda.memory_allocated(dev_idx) / 1024 ** 3
                        log(f"    block[{i:02d}] on cuda:{dev_idx} "
                            f"pre-alloc={al:.2f}GB")
                    return orig(*a, **kw)
                return bfwd

            block.forward = _mk(block.forward, i, 0 if i < split else 1)
    return state


@contextmanager
def oom_guard(tag):
    """On CUDA OOM: dump both GPUs' memory summaries + actionable hints."""
    try:
        yield
    except torch.cuda.OutOfMemoryError:
        log(f"!!! CUDA OOM during: {tag}")
        for i in range(torch.cuda.device_count()):
            print(f"----- torch.cuda.memory_summary(cuda:{i}) -----", flush=True)
            print(torch.cuda.memory_summary(i, abbreviated=True), flush=True)
        log("=================== HOW TO FIX ===================")
        log("The GPU index in the OOM message above tells you the direction:")
        log("  cuda:0 OOM  ->  LOWER  --split_index  (21 -> 20 -> 19 -> 18).")
        log("                  blocks[0:split] live on cuda:0, so RAISING it")
        log("                  (e.g. 24) makes cuda:0 STRICTLY WORSE.")
        log("  cuda:1 OOM  ->  RAISE  --split_index  (20 -> 21 -> 22 -> 23).")
        log("Then: --ffn_chunk_tokens 2400 and --rope_chunk_heads 4 shave the")
        log("activation peaks further. Restart the Kaggle kernel first so the")
        log("notebook process itself holds no VRAM (it held 114MB last run).")
        log("==================================================")
        raise


FP16_MAX = 65504.0
# Clamp activations to a sane magnitude, well below the fp16 ceiling, so a
# single near-overflow value cannot compound across 48 residual additions into
# a true Inf. bf16-trained activations here sit at O(1e2); 1e4 is a generous
# ceiling that preserves signal while breaking runaway growth.
ACT_CLAMP = 1.0e4
_SANITIZE_HITS = {"n": 0, "clamped": 0}


def _sanitize_(t: torch.Tensor, tag: str = "", verbose: bool = False,
               clamp: float = None) -> torch.Tensor:
    """Replace NaN/Inf AND clamp to +/-clamp (in-place), fp16-safe.

    Root cause of the black video: LongCat trains in bf16 (range ~3e38); on T4
    we must use fp16 (max 65504). Across 48 residual blocks, values drift up;
    once one exceeds 65504 it becomes Inf, then NaN, and every latent goes
    black. Clamping to a sane magnitude at each block output keeps the residual
    stream finite without destroying the signal (unlike clamping only at the
    fp16 ceiling, which fires too late).
    """
    if clamp is None:
        clamp = ACT_CLAMP
    if t.dtype == torch.float16:
        # a replacement value above 65504 would cast to fp16 Inf and defeat
        # the scrub; cap it at the fp16 ceiling for fp16 tensors.
        clamp = min(clamp, FP16_MAX)
    if not torch.is_floating_point(t):
        return t
    nonfinite = not bool(torch.isfinite(t).all())
    if nonfinite:
        _SANITIZE_HITS["n"] += 1
        if verbose and _SANITIZE_HITS["n"] <= 60:
            log(f"    !! non-finite in {tag} -> fixed (hit #{_SANITIZE_HITS['n']})")
    torch.nan_to_num_(t, nan=0.0, posinf=clamp, neginf=-clamp)
    t.clamp_(-clamp, clamp)
    return t


def install_sanitizer(model, verbose: bool = True):
    """Wrap every block + final_layer to scrub NaN/Inf from their outputs."""
    def _wrap(module, tag):
        orig = module.forward

        def fwd(*a, **kw):
            out = orig(*a, **kw)
            if torch.is_tensor(out):
                return _sanitize_(out, tag, verbose)
            if isinstance(out, (tuple, list)):
                for o in out:
                    if torch.is_tensor(o):
                        _sanitize_(o, tag, verbose)
            return out
        module.forward = fwd

    for i, block in enumerate(model.blocks):
        _wrap(block, f"blocks[{i}]")
    _wrap(model.final_layer, "final_layer")
    log(f"Activation sanitizer installed on {len(model.blocks)} blocks + final_layer "
        f"(fp16 overflow guard).")


def install_chunked_rope(model, chunk_heads: int):
    """Apply 3D RoPE in head-chunks (exact: cos/sin broadcast over heads).

    In generate_avc, rope runs in fp32 over k_full = current + cached-cond +
    ref tokens; the rotate_half torch.stack alone is a ~600MB fp32 tensor and
    the full chain peaks at ~3GB. Chunking 32 heads into groups of
    `chunk_heads` divides that peak by 32/chunk_heads at negligible cost.
    """
    if chunk_heads <= 0:
        log("RoPE head-chunking DISABLED.")
        return
    n = 0
    for block in model.blocks:
        attn = getattr(block, "attn", None)
        rope = getattr(attn, "rope_3d", None) if attn is not None else None
        if rope is None:
            continue
        n_heads = getattr(attn, "num_heads", None) or getattr(attn, "heads", 32)
        orig = rope.forward

        def _make(orig, n_heads):
            def fwd(q, k, *a, **kw):
                dims = [i for i in range(q.dim() - 1) if q.shape[i] == n_heads]
                if not dims:
                    return orig(q, k, *a, **kw)
                d = dims[-1]
                if q.shape[d] <= chunk_heads or k.shape[d] != n_heads:
                    return orig(q, k, *a, **kw)
                qs, ks = [], []
                for s in range(0, n_heads, chunk_heads):
                    sl = [slice(None)] * q.dim()
                    sl[d] = slice(s, s + chunk_heads)
                    sl = tuple(sl)
                    qo, ko = orig(q[sl], k[sl], *a, **kw)
                    qs.append(qo)
                    ks.append(ko)
                return torch.cat(qs, dim=d), torch.cat(ks, dim=d)
            return fwd

        rope.forward = _make(orig, n_heads)
        n += 1
    log(f"Chunked RoPE installed on {n} attention modules "
        f"({chunk_heads} heads per chunk).")


def rescale_residual_stream_avatar(model, shrink: float = 1.0,
                                   attn_shrink: float = 16.0,
                                   ffn_shrink: float = 16.0,
                                   ffn_hidden_shrink: float = 4.0,
                                   cross_shrink: float = 16.0,
                                   fp32_stream: bool = True):
    """fp16 dynamic-range fix for the avatar DiT, v3. Two modes, both exact.

    MODE A (fp32_stream=True, RECOMMENDED -- emulates the bf16 training range):
      The measured failure (v2 post-mortem): even with S=A=F2=16 folds, every
      LEAF module in blocks[0] stayed finite, yet the BLOCK output went Inf ->
      the overflow happens in the INLINE stream ops (gated residual adds /
      modulate) executed in fp16: the residual-stream VALUES themselves exceed
      65504/16. No global shrink is provably enough because a few
      massive-activation channels (blocks[31]: p99.9=0, outlier-ratio=inf)
      set the absmax. The root fix is to give the stream the same exponent
      range it was trained with: carry it in fp32.

        * x_embedder output -> .float(): the stream is BORN fp32 and every
          `x = x + ...` stays fp32 (adds/mods promote; norms are fp32 inside).
        * every (Quantized)Linear gets a forward-pre cast fp32 -> fp16
          (installed separately by install_fp16_input_cast): all matmul
          inputs come from scale-invariant norms / modulate, i.e. are O(1),
          so the cast is loss-free; matmuls keep full fp16 tensor-core speed.
        * adaLN / audio_adaLN outputs -> .float(): gate * branch and
          modulate then run in fp32 -> no fp16 product overflow, and NO
          chunk-order assumptions are needed (v2 scaled chunk 2/6 and 5/6 --
          correct only if the modulation layout matches; v3 removes that
          assumption entirely in this mode).
        * branch projections still run their MATMULS in fp16 and were
          measured to saturate internally (attn.proj: in 1.88e3 finite ->
          out Inf; ffn.w2: in 1.45e4 -> Inf). Keep the weight folds for
          internal headroom and restore the exact value in fp32 right at the
          module output:
             attn.proj       /= A,  output-hook *= A          (fp32)
             ffn.w3          /= F3  (shrinks the SwiGLU hidden product),
             ffn.w2          /= F2, output-hook *= F2*F3      (fp32)
             cross_attn.proj /= C,  output-hook *= C          (fp32)
             audio_cross_attn.proj /= C, NO hook: its output goes through a
               LayerNorm before the gated add -> scale-invariant, the fold
               is free internal headroom.
          Fold + same-factor fp32 restore at the output is an exact identity
          (LoRA multipliers of folded modules are divided too, so the
          restore covers them as well).
      Memory cost: the hidden-state stream in fp32 (~+0.6 GB peak per GPU at
      480p). Everything heavy (weights int8, matmuls, attention) stays fp16.

    MODE B (fp32_stream=False -- the v2 all-fp16 path, fallback if MODE A
      OOMs): stream carries x/S; branch folds compensated in the adaLN gate
      chunks (assumes 6-chunk [shift,scale,gate]x2 layout, gate_msa=idx 2,
      gate_mlp=idx 5; audio 3-chunk, gate=idx 2). Requires cross_shrink == S.
      Sized from the probe: it suggested x4 on top of 16 -> use S=64.
    """
    S, A, F2, F3, C = (float(shrink), float(attn_shrink), float(ffn_shrink),
                       float(ffn_hidden_shrink), float(cross_shrink))
    assert min(S, A, F2, F3, C) >= 1.0
    handles = []

    # LoRAs wrap module forwards and read lora.multiplier at call time
    # (_create_multi_lora_forward); index them by wrapped-module name.
    lora_by_module = {}
    for key in getattr(model, "active_loras", []):
        net = model.lora_dict.get(key)
        if net is None:
            continue
        for lora in net.loras:
            mname = lora.lora_name.replace("lora___lorahyphen___", "") \
                                  .replace("___lorahyphen___", ".")
            lora_by_module.setdefault(mname, []).append(lora)

    n_folded = n_lora = 0

    def fold(module_name, factor):
        nonlocal n_folded, n_lora
        if factor == 1.0:
            return
        mod = model.get_submodule(module_name)
        assert getattr(mod, "_fp16_folded", None) is None, \
            f"{module_name} already folded"
        assert hasattr(mod, "weight_scale"), \
            f"{module_name} is not a QuantizedLinear; got {type(mod).__name__}"
        with torch.no_grad():
            mod.weight_scale.mul_(1.0 / factor)
            if getattr(mod, "bias", None) is not None:
                mod.bias.mul_(1.0 / factor)
        for lora in lora_by_module.get(module_name, []):
            lora.multiplier = lora.multiplier / factor
            n_lora += 1
        mod._fp16_folded = factor
        n_folded += 1

    def out_to_fp32(mul: float = 1.0):
        def hook(mod, inputs, out):
            return out.float() * mul if mul != 1.0 else out.float()
        return hook

    def scale_out(inv):
        def hook(mod, inputs, out):
            out.mul_(inv)
            return out
        return hook

    def scale_gate_chunks(n_chunks, idx_factor):
        idx_factor = [(g, f) for g, f in idx_factor if f != 1.0]
        if not idx_factor:
            return None
        def hook(mod, inputs, out):
            Cc = out.shape[-1] // n_chunks
            for g, f in idx_factor:
                out[..., g * Cc:(g + 1) * Cc] *= f
            return out
        return hook

    if fp32_stream:
        assert S == 1.0, "fp32 stream needs no shrink; pass shrink=1"
        # stream is born fp32
        handles.append(model.x_embedder.register_forward_hook(out_to_fp32()))
        for i in range(len(model.blocks)):
            blk = model.blocks[i]
            fold(f"blocks.{i}.attn.proj", A)
            fold(f"blocks.{i}.ffn.w3", F3)
            fold(f"blocks.{i}.ffn.w2", F2)
            fold(f"blocks.{i}.cross_attn.proj", C)
            fold(f"blocks.{i}.audio_cross_attn.proj", C)  # norm-invariant
            if A != 1.0:
                handles.append(blk.attn.proj.register_forward_hook(
                    out_to_fp32(A)))
            if F2 * F3 != 1.0:
                handles.append(blk.ffn.w2.register_forward_hook(
                    out_to_fp32(F2 * F3)))
            if C != 1.0:
                handles.append(blk.cross_attn.proj.register_forward_hook(
                    out_to_fp32(C)))
            # gates / modulate in fp32: cast the whole adaLN output, no
            # chunk-layout assumption.
            handles.append(blk.adaLN_modulation.register_forward_hook(
                out_to_fp32()))
            handles.append(blk.audio_adaLN_modulation.register_forward_hook(
                out_to_fp32()))
        log(f"fp16 range fix v3 [fp32 stream]: x_embedder->fp32, adaLN->fp32, "
            f"branch folds attn.proj/{A:g} ffn.w2/{F2:g} ffn.w3/{F3:g} "
            f"cross.proj/{C:g} audio.proj/{C:g} with exact fp32 restore "
            f"({n_folded} folds, {n_lora} LoRA multipliers compensated, "
            f"{len(handles)} hooks). Stream now has bf16-equivalent range; "
            f"matmuls stay fp16.")
        return handles

    # ---- MODE B: all-fp16 stream shrink (v2 behaviour, bigger defaults) ----
    assert C == S, ("fp16-stream mode requires --cross_shrink == "
                    "--residual_shrink (raw ungated residual add).")
    if S > 1.0:
        handles.append(model.x_embedder.register_forward_hook(
            scale_out(1.0 / S)))
    for i in range(len(model.blocks)):
        blk = model.blocks[i]
        fold(f"blocks.{i}.attn.proj", A)
        fold(f"blocks.{i}.ffn.w3", F3)
        fold(f"blocks.{i}.ffn.w2", F2)
        fold(f"blocks.{i}.cross_attn.proj", S)
        fold(f"blocks.{i}.audio_cross_attn.proj", A)  # norm-invariant
        h = scale_gate_chunks(6, [(2, A / S), (5, F2 * F3 / S)])
        if h is not None:
            handles.append(blk.adaLN_modulation.register_forward_hook(h))
        h = scale_gate_chunks(3, [(2, 1.0 / S)])
        if h is not None:
            handles.append(blk.audio_adaLN_modulation.register_forward_hook(h))
    log(f"fp16 range fix v3 [fp16 stream/{S:g}] (+{math.log2(S):.1f} bits), "
        f"branch folds attn.proj/{A:g} ffn.w2/{F2:g} ffn.w3/{F3:g} "
        f"cross_attn.proj/{S:g} ({n_folded} projections folded, {n_lora} "
        f"LoRA multipliers compensated, {len(handles)} hooks). All "
        f"transforms are exact identities.")
    return handles


def install_fp16_input_cast(model):
    """Cast fp32 tensor inputs -> fp16 at every (Quantized)Linear.

    Companion of the fp32-stream mode: all Linear inputs come from
    scale-invariant norms / fp32 modulate (values O(1)), so the cast is
    loss-free while keeping the matmuls in fp16. Registered as forward-PRE
    hooks so they fire before any LoRA-wrapped forward.
    """
    def pre(mod, args, kwargs):
        args = tuple(
            a.half() if torch.is_tensor(a) and a.dtype == torch.float32 else a
            for a in args)
        kwargs = {
            k: (v.half() if torch.is_tensor(v) and v.dtype == torch.float32
                else v)
            for k, v in kwargs.items()}
        return args, kwargs

    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, (QuantizedLinear, nn.Linear)):
            mod.register_forward_pre_hook(pre, with_kwargs=True)
            n += 1
    log(f"fp32->fp16 input cast installed on {n} (Quantized)Linear modules.")
    return n


def load_int8_dit_pipeline_parallel(
    checkpoint_dir: str,
    subfolder: str = "base_model_int8",
    split_index: int = 22,
    compute_dtype: torch.dtype = torch.float16,
    cp_split_hw=(1, 1),
):
    qdir = os.path.join(checkpoint_dir, subfolder)
    with open(os.path.join(qdir, "config.json"), "r") as f:
        config = json.load(f)
    for k in ("_class_name", "architectures", "_diffusers_version", "model_max_length"):
        config.pop(k, None)

    # T4: no flash-attn / no BSA(triton on turing is unreliable) -> xformers.
    config.update(
        dict(
            enable_flashattn2=False,
            enable_flashattn3=False,
            enable_xformers=True,
            enable_bsa=False,
            cp_split_hw=list(cp_split_hw),
        )
    )
    depth = int(config.get("depth", 48))
    assert 0 < split_index < depth, f"--split_index must be in (0, {depth})"
    log(f"Instantiating DiT on meta device (depth={depth}, split at block {split_index}) ...")

    # -- 1) build the model with ZERO memory: params on meta ------------------
    with init_empty_weights(include_buffers=False):
        model = LongCatVideoAvatarTransformer3DModel(**config)

    # -- 2) replace nn.Linear -> QuantizedLinear, buffers also on meta --------
    to_replace = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not any(p in name for p in DEFAULT_SKIP_PATTERNS):
            to_replace[name] = module
    with init_empty_weights(include_buffers=True):
        for name, lin in to_replace.items():
            ql = QuantizedLinear(lin.in_features, lin.out_features, bias=lin.bias is not None)
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], ql)
    log(f"Replaced {len(to_replace)} Linear layers with QuantizedLinear (meta).")

    # -- 3) stream shards straight to their target GPUs -----------------------
    index_path = os.path.join(qdir, "quantized_model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shard_files = sorted(set(weight_map.values()))
    else:
        shard_files = sorted(
            f for f in os.listdir(qdir) if f.endswith(".safetensors") and "index" not in f
        )
    assert shard_files, f"No safetensors shards found under {qdir}"

    loaded_keys = set()
    for shard in shard_files:
        t0 = time.time()
        sd = load_file(os.path.join(qdir, shard), device="cpu")
        placed = {}
        for k, v in sd.items():
            dev = _key_device(k, split_index)
            if v.dtype in (torch.float32, torch.bfloat16, torch.float16):
                if "weight_scale" in k:
                    placed[k] = v.to(dev)  # keep fp32 scales for accuracy
                else:
                    placed[k] = v.to(dev, dtype=compute_dtype)
            else:  # int8 weights
                placed[k] = v.to(dev)
        missing, unexpected = model.load_state_dict(placed, strict=False, assign=True)
        if unexpected:
            log(f"WARNING: unexpected keys in {shard} (first 5): {unexpected[:5]}")
        loaded_keys.update(placed.keys())
        del sd, placed
        gc.collect()
        log(f"  shard {shard} -> GPUs in {time.time() - t0:.1f}s")

    # -- 4) sanity: nothing left on meta; move stray CPU buffers --------------
    meta_left = [n for n, p in model.named_parameters() if p.is_meta]
    meta_left += [n for n, b in model.named_buffers() if b is not None and b.is_meta]
    if meta_left:
        raise RuntimeError(
            "Checkpoint did not cover these tensors (still on meta): "
            + ", ".join(meta_left[:10])
        )
    for n, b in model.named_buffers():
        if b is not None and b.device.type == "cpu":
            dev = _key_device(n, split_index)
            _assign_buffer(model, n, b.to(dev))
    for n, p in model.named_parameters():
        if p.device.type == "cpu":  # should not happen, but be safe
            dev = _key_device(n, split_index)
            p.data = p.data.to(dev, dtype=compute_dtype)

    model.eval()
    model.requires_grad_(False)

    # -- 4b) assert each block is fully on its target device -----------------
    for i, block in enumerate(model.blocks):
        want = DEV0 if i < split_index else DEV1
        devs = {p.device.type + ":" + str(p.device.index) for p in block.parameters()}
        devs |= {b.device.type + ":" + str(b.device.index)
                 for b in block.buffers() if b is not None}
        bad = {d for d in devs if d != want}
        if bad:
            log(f"WARNING: block[{i}] has params on {bad}, expected {want}; "
                f"forcing move.")
            block.to(want)

    # -- 5) install device-routing wrappers ----------------------------------
    for i, block in enumerate(model.blocks):
        _wrap_inputs_to_device(block, DEV0 if i < split_index else DEV1)
    _wrap_inputs_to_device(model.final_layer, DEV1)

    main_device = torch.device(DEV0)
    orig_forward = model.forward

    def _routed_forward(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        if isinstance(out, tuple):
            return (out[0].to(main_device),) + tuple(out[1:])
        return out.to(main_device)

    model.forward = _routed_forward
    model._t4pp_split_index = split_index
    return model


def _assign_buffer(model: nn.Module, full_name: str, tensor: torch.Tensor):
    parts = full_name.split(".")
    mod = model
    for p in parts[:-1]:
        mod = getattr(mod, p)
    setattr(mod, parts[-1], tensor)


def redistribute_loras(dit: LongCatVideoAvatarTransformer3DModel):
    """enable_loras() moves every LoRA to the FIRST param's device (cuda:0).

    LoRA forward only casts dtype, not device, so LoRAs attached to modules
    living on cuda:1 would crash. Move each LoRA next to its wrapped module.
    """
    moved = 0
    for key in list(dit.active_loras):
        network = dit.lora_dict.get(key)
        if network is None:
            continue
        for lora in network.loras:
            module_name = (
                lora.lora_name.replace("lora___lorahyphen___", "").replace("___lorahyphen___", ".")
            )
            try:
                module = dit._get_module_by_name(module_name)
            except Exception:
                continue
            dev = None
            for p in module.parameters(recurse=False):
                dev = p.device
                break
            if dev is None:
                for b in module.buffers(recurse=False):
                    if b is not None:
                        dev = b.device
                        break
            if dev is not None:
                lora.to(dev)
                moved += 1
    log(f"Redistributed {moved} LoRA modules to their host devices.")


# ---------------------------------------------------------------------------
# 2. One-shot encoders: audio embedding + prompt embedding, then free
# ---------------------------------------------------------------------------
def compute_full_audio_embedding(pipe, checkpoint_dir, model_type, speech_array, sr, save_fps, audio_stride):
    if model_type == "avatar-v1.0":
        audio_model_path = os.path.join(checkpoint_dir, "chinese-wav2vec2-base")
    else:
        audio_model_path = os.path.join(checkpoint_dir, "whisper-large-v3")

    log(f"Loading audio encoder from {audio_model_path} ...")
    audio_encoder = get_audio_encoder(audio_model_path, model_type)
    try:
        audio_encoder = audio_encoder.to(dtype=torch.float16)
    except Exception:
        pass
    audio_encoder = audio_encoder.to(DEV0).eval()
    audio_feature_extractor = get_audio_feature_extractor(audio_model_path, model_type)

    pipe.audio_encoder = audio_encoder
    pipe.audio_feature_extractor = audio_feature_extractor

    with torch.no_grad():
        full_audio_emb = pipe.get_audio_embedding(
            speech_array, fps=save_fps * audio_stride, device=DEV0,
            sample_rate=sr, model_type=model_type,
        )

    # Keep the long full-audio embedding on CPU. For an ~82s clip this saves
    # ~50 MiB on cuda:0; each segment copies only its 93-frame audio window.
    full_audio_emb = torch.nan_to_num(
        full_audio_emb.to("cpu", dtype=torch.float32),
        nan=0.0, posinf=FP16_MAX, neginf=-FP16_MAX,
    )
    if torch.isnan(full_audio_emb).any():
        raise ValueError("Broken audio embedding: NaN values detected.")

    # free the encoder
    pipe.audio_encoder = None
    del audio_encoder
    torch_gc()
    log(f"Audio embedding ready: {tuple(full_audio_emb.shape)}; encoder freed.")
    return full_audio_emb


def encode_and_freeze_prompts(pipe, base_dir, prompt, negative_prompt, dtype):
    log("Loading UMT5-XXL text encoder (fp16, one-shot) ...")
    text_encoder = UMT5EncoderModel.from_pretrained(
        base_dir, subfolder="text_encoder", torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEV0).eval()
    pipe.text_encoder = text_encoder

    with torch.no_grad():
        pe, pm, npe, npm = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,  # cache both branches
            device=DEV0,
            dtype=dtype,
        )
    def _scrub_emb(t, tag):
        if t is None:
            return None
        bad = int((~torch.isfinite(t)).sum())
        am = torch.nan_to_num(t.float(), nan=0.0, posinf=0.0,
                              neginf=0.0).abs().max().item()
        log(f"prompt emb[{tag}]: shape={tuple(t.shape)} "
            f"absmax(finite)={am:.4g} nonfinite={bad}")
        if bad:
            log(f"WARNING: prompt emb[{tag}] has {bad} NaN/Inf values from "
                f"the fp16 UMT5 pass -- scrubbed, but if quality is still "
                f"bad the text encoder needs a higher-precision pass.")
        return torch.nan_to_num(t, nan=0.0, posinf=FP16_MAX, neginf=-FP16_MAX)

    pe, npe = _scrub_emb(pe, "pos"), _scrub_emb(npe, "neg")
    d_model = text_encoder.config.d_model

    pipe.text_encoder = None
    del text_encoder
    torch_gc()
    log("Prompt embeddings cached; text encoder freed.")

    cache = (pe, pm, npe, npm)

    def cached_encode_prompt(prompt=None, negative_prompt=None,
                             do_classifier_free_guidance=True, **kwargs):
        if do_classifier_free_guidance:
            return cache
        return cache[0], cache[1], None, None

    pipe.encode_prompt = cached_encode_prompt
    # lightweight stub: _cache_clean_latents reads text_encoder.config.d_model
    pipe.text_encoder = types.SimpleNamespace(
        config=types.SimpleNamespace(d_model=d_model), dtype=torch.float16
    )


# ---------------------------------------------------------------------------
# 3. Audio preparation (vocal separation + padding)
# ---------------------------------------------------------------------------
def generate_random_uid():
    return str(int(time.time()))[-6:] + str(random.randint(100000, 999999))


def extract_vocal(source_path, checkpoint_dir):
    vocal_model_path = os.path.join(checkpoint_dir, "vocal_separator", "Kim_Vocal_2.onnx")
    tmp_dir = Path("./audio_temp_file")
    (tmp_dir / "vocals").mkdir(parents=True, exist_ok=True)
    separator = Separator(
        output_dir=tmp_dir / "vocals",
        output_single_stem="vocals",
        model_file_dir=os.path.dirname(vocal_model_path),
    )
    separator.load_model(os.path.basename(vocal_model_path))
    outputs = separator.separate(source_path)
    del separator
    gc.collect()
    if not outputs:
        log("Vocal separation produced nothing; using raw audio.")
        return source_path
    vocal_path = (tmp_dir / "vocals" / outputs[0]).resolve().as_posix()
    target = os.path.abspath(f"temp_speech_{generate_random_uid()}_vocal.wav")
    os.replace(vocal_path, target)
    return target


# ---------------------------------------------------------------------------
# 4. Incremental video writing (constant CPU memory over segments)
# ---------------------------------------------------------------------------
def write_segment_video(frames, path, fps):
    import imageio
    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", quality=8, macro_block_size=1,
        pixelformat="yuv420p",
    )
    for f in frames:
        writer.append_data(np.asarray(f))
    writer.close()


def concat_and_mux(segment_paths, audio_path, out_path):
    list_file = out_path + ".concat.txt"
    with open(list_file, "w") as f:
        for p in segment_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
        "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-shortest", out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(list_file)
    log(f"Final video written: {out_path}")


def _save_resume_state(output_dir, segment_idx, latent, ref_latent,
                       current_video, audio_start_idx, generator,
                       segment_paths):
    """Atomically persist everything the continuation loop needs.

    ~120 MB (uint8 frames dominate). Written after every segment so a killed
    12h Kaggle session loses at most one segment of work."""
    state = {
        "segment_idx": segment_idx,       # segments completed so far (1-based)
        "latent": latent.detach().to("cpu", torch.float16),
        "ref_latent": ref_latent.detach().to("cpu", torch.float16),
        "frames": np.stack([np.asarray(f, dtype=np.uint8)
                            for f in current_video]),
        "audio_start_idx": int(audio_start_idx),
        "generator_state": generator.get_state(),
        "segment_paths": list(segment_paths),
    }
    path = os.path.join(output_dir, "resume_state.pt")
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)
    log(f"  resume state saved (after segment {segment_idx}).")


def _load_resume_state(output_dir):
    path = os.path.join(output_dir, "resume_state.pt")
    if not os.path.exists(path):
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        missing = [p for p in state["segment_paths"] if not os.path.exists(p)]
        if missing:
            log(f"RESUME: segment files missing ({missing[:3]} ...); "
                f"starting fresh.")
            return None
        log(f"RESUME: state found -- {state['segment_idx']} segment(s) "
            f"already done, continuing.")
        return state
    except Exception as e:
        log(f"RESUME: failed to load state ({e}); starting fresh.")
        return None


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def generate(args):
    assert torch.cuda.device_count() >= 2, "This script needs 2 GPUs (Kaggle: GPU T4 x2)."
    torch.cuda.set_device(0)

    # single-process "distributed" env so context_parallel_util is happy
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29512")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    context_parallel_util.init_context_parallel(
        context_parallel_size=1, global_rank=0, world_size=1
    )
    cp_split_hw = context_parallel_util.get_optimal_split(1)

    checkpoint_dir = args.checkpoint_dir
    base_dir = args.longcat_video_dir or os.path.join(checkpoint_dir, "..", "LongCat-Video")
    model_type = args.model_type
    use_distill = args.use_distill
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if model_type != "avatar-v1.5":
        raise ValueError(
            "Only avatar-v1.5 fits Kaggle T4s: it is the only version with an official "
            "INT8 DiT and an 8-step distilled sampler. Use --model_type avatar-v1.5."
        )
    if not args.use_int8:
        raise ValueError("bf16/fp16 full weights (27GB) cannot fit 2x15GB T4s. Keep --use_int8.")

    num_inference_steps = args.num_inference_steps
    text_guidance_scale = args.text_guidance_scale
    audio_guidance_scale = args.audio_guidance_scale
    if use_distill:
        num_inference_steps = 8
        text_guidance_scale = 1.0
        audio_guidance_scale = 1.0
        log("Distill mode: 8 steps, CFG disabled (1 forward pass per step).")

    # v1.5 constants
    save_fps = 25
    audio_stride = 1
    num_frames = 93
    num_cond_frames = 13

    if args.resolution == "480p":
        height, width = 480, 832
    else:
        raise ValueError("720p does not fit T4 VRAM with this model; use 480p.")

    with open(args.input_json, "r", encoding="utf-8") as f:
        input_data = json.load(f)
    prompt = input_data["prompt"]
    raw_speech_path = input_data["cond_audio"]["person1"]
    if args.max_audio_seconds > 0:
        trimmed = os.path.abspath(f"trimmed_input_{generate_random_uid()}.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", raw_speech_path,
             "-t", str(args.max_audio_seconds), "-ar", "16000", "-ac", "1", trimmed],
            check=True,
        )
        log(f"MVP mode: audio trimmed to first {args.max_audio_seconds}s -> {trimmed}")
        raw_speech_path = trimmed

    # ---- audio: vocal separation, duration, padding -------------------------
    log("Separating vocals (CPU/onnx) ...")
    vocal_path = extract_vocal(raw_speech_path, checkpoint_dir)
    speech_array, sr = librosa.load(vocal_path, sr=16000)
    source_duration = len(speech_array) / sr
    log(f"Vocal track: {source_duration:.2f}s")

    # how many segments can the audio actually drive?
    seg_new = (num_frames - num_cond_frames) / save_fps      # 3.2 s per extra segment
    seg_first = num_frames / save_fps                        # 3.72 s for segment 1
    max_segments = max(1, 1 + int(math.floor((source_duration - seg_first) / seg_new + 1e-6)) + 0)
    if args.num_segments == "auto":
        num_segments = max_segments
    else:
        num_segments = max(1, int(args.num_segments))
    log(f"num_segments={num_segments} (audio supports up to ~{max_segments}); "
        f"video length ~= {seg_first + (num_segments - 1) * seg_new:.1f}s")

    generate_duration = seg_first + (num_segments - 1) * seg_new
    pad = math.ceil((generate_duration - source_duration) * sr)
    if pad > 0:
        speech_array = np.append(speech_array, np.zeros(pad, dtype=speech_array.dtype))

    # ---- pipeline skeleton (DiT attached later) -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(base_dir, subfolder="tokenizer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler")
    log("Loading VAE (fp16 -> cuda:0) ...")
    vae = AutoencoderKLWan.from_pretrained(
        base_dir, subfolder="vae", torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEV0).eval()
    # Best-effort: enable tiled/sliced VAE decode so the final decode does not
    # spike cuda:0 (it shares the GPU with 16 DiT blocks). Not all builds expose
    # these; guard each call.
    for meth in ("enable_tiling", "enable_slicing"):
        fn = getattr(vae, meth, None)
        if callable(fn):
            try:
                fn()
                log(f"VAE {meth}() enabled.")
            except Exception as e:
                log(f"VAE {meth}() unavailable: {e}")

    pipe = LongCatVideoAvatarPipeline(
        tokenizer=tokenizer,
        text_encoder=None,       # attached transiently below
        vae=vae,
        scheduler=scheduler,
        dit=None,                # attached after encoders are freed
        audio_encoder=None,
        audio_feature_extractor=None,
        model_type=model_type,
    )
    pipe.device = DEV0
    pipe._t4pp_offload_vae_during_denoise = not args.no_vae_offload
    pipe._t4pp_teacache_lite = bool(args.teacache_lite)
    pipe._t4pp_teacache_skip_steps = args.teacache_skip_steps
    pipe._t4pp_teacache_cache_device = args.teacache_cache_device
    log(f"VAE offload during DiT denoise: {pipe._t4pp_offload_vae_during_denoise}")
    if pipe._t4pp_teacache_lite:
        log(f"TeaCache-lite enabled: skip_steps={args.teacache_skip_steps}, cache={args.teacache_cache_device}")

    # ---- phase A: whisper -> full audio embedding, then free ----------------
    full_audio_emb = compute_full_audio_embedding(
        pipe, checkpoint_dir, model_type, speech_array, sr, save_fps, audio_stride
    )
    full_audio_emb = torch.nan_to_num(full_audio_emb, nan=0.0, posinf=FP16_MAX, neginf=-FP16_MAX).to("cpu", torch.float32)
    torch_gc()
    report_vram("after-audio")

    # ---- phase B: UMT5 -> cached prompt embeddings, then free ---------------
    encode_and_freeze_prompts(pipe, base_dir, prompt, NEGATIVE_PROMPT, dtype=torch.float16)
    report_vram("after-text")

    # ---- phase C: INT8 DiT, split across both GPUs ---------------------------
    dit = load_int8_dit_pipeline_parallel(
        checkpoint_dir,
        subfolder="base_model_int8",
        split_index=args.split_index,
        compute_dtype=torch.float16,
        cp_split_hw=cp_split_hw,
    )
    if use_distill:
        lora_path = os.path.join(checkpoint_dir, "lora", "dmd_lora.safetensors")
        if os.path.exists(lora_path):
            log("Loading DMD distillation LoRA ...")
            dit.load_lora(lora_path, "dmd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
            dit.enable_loras(["dmd"])
            redistribute_loras(dit)
        else:
            log(f"WARNING: {lora_path} not found; distill sampling without LoRA will look bad.")
    install_chunked_ffn(dit, args.ffn_chunk_tokens)
    install_chunked_rope(dit, args.rope_chunk_heads)

    # ---- fp16 dynamic-range: tracer (locate) + rescale/fold (fix) + probe ---
    if args.fp16_trace_nonfinite:
        # BEFORE the clamp hooks, so block-level hits are seen unscrubbed.
        install_nonfinite_tracer(dit, max_reports=40, log_fn=log)
    fp32_stream = (args.stream_dtype == "float32")
    if fp32_stream and args.residual_shrink != 1.0:
        log(f"stream_dtype=float32 -> ignoring --residual_shrink "
            f"{args.residual_shrink:g} (fp32 stream needs no shrink).")
        args.residual_shrink = 1.0
    if not fp32_stream and args.cross_shrink != args.residual_shrink:
        log(f"stream_dtype=float16 -> --cross_shrink must equal "
            f"--residual_shrink (ungated residual add); forcing "
            f"{args.residual_shrink:g}.")
        args.cross_shrink = args.residual_shrink
    rescale_residual_stream_avatar(
        dit, shrink=args.residual_shrink, attn_shrink=args.attn_shrink,
        ffn_shrink=args.ffn_shrink,
        ffn_hidden_shrink=args.ffn_hidden_shrink,
        cross_shrink=args.cross_shrink, fp32_stream=fp32_stream)
    if fp32_stream:
        install_fp16_input_cast(dit)
        if not args.no_soft_clamp:
            # fp32 stream may LEGALLY exceed 65504 -> never magnitude-clamp;
            # keep only the NaN/Inf sentinel (should never fire now).
            install_soft_clamp(dit, limit=args.soft_clamp_limit, log_fn=log,
                               magnitude_clamp=False)
        # sanitizer degrades to a pure NaN/Inf scrubber with an fp32 ceiling
        globals()["ACT_CLAMP"] = 3.0e38
    else:
        if not args.no_soft_clamp:
            install_soft_clamp(dit, limit=args.soft_clamp_limit, log_fn=log)
        # With headroom restored the sanitizer degrades to a NaN/Inf scrubber.
        globals()["ACT_CLAMP"] = FP16_MAX
    if args.fp16_probe:
        print_block_anatomy(dit)
        fp16_stats = install_block_stats(dit)
        fp16_csv = os.path.join(output_dir, "fp16_stats.csv")

        def _dump_fp16_stats():
            try:
                fp16_stats.summary()
                fp16_stats.to_csv(fp16_csv)
                s = fp16_stats.suggest_shrink(target_bits=2.0)
                log(f"suggested EXTRA shrink on top of --residual_shrink "
                    f"{args.residual_shrink:g}: x{s:g}")
            except Exception as e:
                log(f"fp16 stats dump failed: {e}")

        atexit.register(_dump_fp16_stats)
    if not args.no_sanitize:
        install_sanitizer(dit, verbose=args.debug_level >= 1)
    debug_state = install_dit_debug(dit, level=args.debug_level,
                                    empty_cache=not args.no_empty_cache)
    pipe.dit = dit
    torch_gc()
    report_vram("after-dit")

    generator = torch.Generator(device=DEV0)
    generator.manual_seed(args.seed)

    # ---- first-clip audio window --------------------------------------------
    indices = torch.arange(2 * 2 + 1) - 2
    audio_start_idx = 0
    audio_end_idx = audio_start_idx + audio_stride * num_frames
    center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
    audio_emb = full_audio_emb[center_indices][None, ...].to(DEV0, non_blocking=True)

    segment_paths = []
    resume_state = _load_resume_state(output_dir) if args.resume else None

    if resume_state is not None:
        # -- restore everything the continuation loop needs -------------------
        segments_done = int(resume_state["segment_idx"])
        latent = resume_state["latent"].to(DEV0, torch.float16)
        ref_latent = resume_state["ref_latent"].to(DEV0, torch.float16)
        current_video = [PIL.Image.fromarray(f)
                         for f in resume_state["frames"]]
        audio_start_idx = resume_state["audio_start_idx"]
        generator.set_state(resume_state["generator_state"])
        segment_paths = list(resume_state["segment_paths"])
        width_px, height_px = current_video[0].size
        log(f"RESUME: continuing from segment {segments_done + 1}/{num_segments}.")
    else:
        segments_done = 1
        t_seg = time.time()
        log(f"Generating segment 1/{num_segments} ({args.stage_1}) ...")

        common = dict(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            generator=generator,
            output_type="both",
            audio_emb=audio_emb,
            use_distill=use_distill,
        )
        reset_peaks()
        with oom_guard(f"segment 1/{num_segments} ({args.stage_1})"):
            if args.stage_1 == "at2v":
                output, latent = pipe.generate_at2v(height=height, width=width, **common)
            elif args.stage_1 == "ai2v":
                image = load_image(input_data["cond_image"])
                output, latent = pipe.generate_ai2v(image=image, resolution=args.resolution, **common)
            else:
                raise NotImplementedError(args.stage_1)
        report_peaks("segment-1")

        output = output[0]
        _pixel_report(output, "segment-1")
        video = [PIL.Image.fromarray((output[i] * 255).astype(np.uint8)) for i in range(output.shape[0])]
        del output
        torch_gc()
        log(f"Segment 1 done in {(time.time() - t_seg) / 60:.1f} min.")

        seg_path = os.path.join(output_dir, "segment_0001.mp4")
        write_segment_video(video, seg_path, save_fps)
        segment_paths.append(seg_path)

        width_px, height_px = video[0].size
        current_video = video
        latent = torch.nan_to_num(latent, nan=0.0, posinf=FP16_MAX, neginf=-FP16_MAX)
        ref_latent = latent[:, :, :1].clone()
        if args.resume:
            _save_resume_state(output_dir, 1, latent, ref_latent,
                               current_video, audio_start_idx, generator,
                               segment_paths)

    # ---- long-video continuation loop ----------------------------------------
    for segment_idx in range(segments_done, num_segments):
        t_seg = time.time()
        log(f"Generating segment {segment_idx + 1}/{num_segments} ...")

        audio_start_idx = audio_start_idx + audio_stride * (num_frames - num_cond_frames)
        audio_end_idx = audio_start_idx + audio_stride * num_frames
        center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
        audio_emb = full_audio_emb[center_indices][None, ...].to(DEV0, non_blocking=True)

        log(f"  audio window: frames [{audio_start_idx}, {audio_end_idx}) of {full_audio_emb.shape[0]}")
        reset_peaks()
        with oom_guard(f"segment {segment_idx + 1}/{num_segments} (avc)"):
            output, latent = pipe.generate_avc(
                video=current_video,
                video_latent=latent,
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                height=height_px,
                width=width_px,
                num_frames=num_frames,
                num_cond_frames=num_cond_frames,
                num_inference_steps=num_inference_steps,
                text_guidance_scale=text_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                generator=generator,
                output_type="both",
                use_kv_cache=True,
                offload_kv_cache=args.offload_kv_cache,
                enhance_hf=False if use_distill else True,
                audio_emb=audio_emb,
                ref_latent=ref_latent,
                ref_img_index=args.ref_img_index,
                mask_frame_range=args.mask_frame_range,
                use_distill=use_distill,
            )
        report_peaks(f"segment-{segment_idx + 1}")
        output = output[0]
        _pixel_report(output, f"segment-{segment_idx + 1}")
        latent = torch.nan_to_num(latent, nan=0.0, posinf=FP16_MAX, neginf=-FP16_MAX)
        new_video = [PIL.Image.fromarray((output[i] * 255).astype(np.uint8)) for i in range(output.shape[0])]
        del output

        seg_path = os.path.join(output_dir, f"segment_{segment_idx + 1:04d}.mp4")
        write_segment_video(new_video[num_cond_frames:], seg_path, save_fps)
        segment_paths.append(seg_path)

        current_video = new_video
        torch_gc()
        log(f"Segment {segment_idx + 1} done in {(time.time() - t_seg) / 60:.1f} min.")
        if args.resume:
            _save_resume_state(output_dir, segment_idx + 1, latent, ref_latent,
                               current_video, audio_start_idx, generator,
                               segment_paths)

        # refresh the muxed result every segment so partial progress is usable
        try:
            concat_and_mux(segment_paths, raw_speech_path,
                           os.path.join(output_dir, "final_video.mp4"))
        except subprocess.CalledProcessError:
            log("WARNING: intermediate concat failed (will retry at the end).")

    if len(segment_paths) >= 1:
        concat_and_mux(segment_paths, raw_speech_path, os.path.join(output_dir, "final_video.mp4"))
    log("All segments complete.")
    report_vram("final")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_json", type=str, default="assets/avatar/single_example_1.json")
    p.add_argument("--output_dir", type=str, default="./outputs_avatar_t4")
    p.add_argument("--checkpoint_dir", type=str, default="./weights/LongCat-Video-Avatar-1.5")
    p.add_argument("--longcat_video_dir", type=str, default=None,
                   help="Path to base LongCat-Video weights (tokenizer/text_encoder/vae). "
                        "Defaults to <checkpoint_dir>/../LongCat-Video")
    p.add_argument("--stage_1", type=str, default="ai2v", choices=["ai2v", "at2v"])
    p.add_argument("--resolution", type=str, default="480p", choices=["480p", "720p"])
    p.add_argument("--num_segments", type=str, default="auto",
                   help="'auto' = as many segments as the audio supports, or an integer.")
    p.add_argument("--num_inference_steps", type=int, default=8)
    p.add_argument("--text_guidance_scale", type=float, default=1.0)
    p.add_argument("--audio_guidance_scale", type=float, default=1.0)
    p.add_argument("--ref_img_index", type=int, default=10)
    p.add_argument("--mask_frame_range", type=int, default=3)
    p.add_argument("--model_type", type=str, default="avatar-v1.5")
    p.add_argument("--use_distill", action="store_true", default=True)
    p.add_argument("--no_distill", dest="use_distill", action="store_false")
    p.add_argument("--use_int8", action="store_true", default=True)
    p.add_argument("--max_audio_seconds", type=float, default=0,
                   help="MVP mode: use only the first N seconds of the input audio "
                        "(trim happens before vocal separation; final mux uses the "
                        "trimmed audio). 0 = full length.")
    p.add_argument("--rope_chunk_heads", type=int, default=4,
                   help="Apply 3D RoPE in head-chunks (exact). Main fix for the "
                        "segment-2 (avc/kv-cache) OOM. 0 disables.")
    p.add_argument("--no_sanitize", action="store_true",
                   help="Disable the per-block NaN/Inf sanitizer (NOT recommended "
                        "on T4 fp16 — this is the black-video fix).")
    p.add_argument("--split_index", type=int, default=21,
                   help="blocks[0:split] -> cuda:0, blocks[split:] -> cuda:1. "
                        "Lower it if cuda:0 OOMs, raise it if cuda:1 OOMs. "
                        "cuda:0 also hosts embedders/VAE/latents, so it gets "
                        "fewer blocks by default.")
    p.add_argument("--ffn_chunk_tokens", type=int, default=2400,
                   help="Sequential FFN chunk size in tokens (exact, pointwise). "
                        "Lower = less VRAM, slightly slower. 0 disables.")
    p.add_argument("--debug_level", type=int, default=1, choices=[0, 1, 2],
                   help="0=quiet, 1=per-DiT-forward VRAM heartbeat, "
                        "2=also per-block watermarks on first 2 forwards.")
    p.add_argument("--no_empty_cache", action="store_true",
                   help="Skip empty_cache after each DiT forward (slightly faster, riskier).")
    p.add_argument("--offload_kv_cache", action="store_true", default=True)
    p.add_argument("--keep_kv_on_gpu", dest="offload_kv_cache", action="store_false")
    p.add_argument("--no_vae_offload", action="store_true",
                   help="Keep VAE on cuda:0 during DiT denoising. Default is to offload it to CPU after VAE encode and reload before decode; disabling this is faster but much riskier on 15GB T4.")
    p.add_argument("--teacache_lite", action="store_true",
                   help="Experimental speed mode: reuse cached noise_pred on selected denoise steps. It is off by default because it is approximate and can affect lip/identity quality.")
    p.add_argument("--teacache_skip_steps", type=str, default="2,3,5,6",
                   help="1-based denoise steps to skip when --teacache_lite is enabled. First and last steps are always forced full. Example: '2,4,6' for more conservative speedup.")
    p.add_argument("--teacache_cache_device", type=str, default="cpu", choices=["cpu", "gpu"],
                   help="Where to keep the tiny cached noise_pred. CPU is default and adds almost no GPU memory pressure.")
    p.add_argument("--stream_dtype", type=str, default="float32",
                   choices=["float32", "float16"],
                   help="float32 (default): carry the DiT residual stream in "
                        "fp32 -- same exponent range as bf16 training; the "
                        "root fix for the Inf/花屏 failure. Matmuls stay "
                        "fp16/int8. float16: legacy all-fp16 path, needs big "
                        "--residual_shrink (>=64) and is not robust to "
                        "massive-activation channels.")
    p.add_argument("--cross_shrink", type=float, default=16.0,
                   help="Fold factor for cross_attn.proj / audio_cross_attn"
                        ".proj (internal fp16 headroom, exactly restored in "
                        "fp32). In --stream_dtype float16 mode it must equal "
                        "--residual_shrink.")
    p.add_argument("--no_soft_clamp", action="store_true",
                   help="Skip the per-block sentinel (saves 48 GPU syncs per "
                        "forward, ~2-4%% faster). Only after a verified-clean "
                        "run.")
    p.add_argument("--resume", action="store_true",
                   help="Resume a long run from <output_dir>/resume_state.pt "
                        "(written after every segment). Essential for "
                        "~1-minute videos on a 12h Kaggle session.")
    p.add_argument("--residual_shrink", type=float, default=1.0,
                   help="Exact static rescale of the DiT residual stream by "
                        "1/shrink. Entry norms are scale-invariant so the "
                        "output is unchanged; the fp16 stream gains "
                        "log2(shrink) bits of headroom. 1 disables.")
    p.add_argument("--attn_shrink", type=float, default=16.0,
                   help="Fold 1/s into attn.proj weight scales (+bias+LoRA) "
                        "and multiply gate_msa by s: kills the measured fp16 "
                        "saturation INSIDE the self-attn output projection. "
                        "Exact. 1 disables.")
    p.add_argument("--ffn_shrink", type=float, default=16.0,
                   help="Same folding for ffn.w2 (down-projection), "
                        "compensated in gate_mlp. Exact. 1 disables.")
    p.add_argument("--ffn_hidden_shrink", type=float, default=4.0,
                   help="Fold 1/s into ffn.w3 to shrink the SwiGLU hidden "
                        "activation (measured 1.45e4, only 2 bits from "
                        "overflow); silu sits on w1 so w3 is the linear-safe "
                        "factor. Also compensated in gate_mlp. 1 disables.")
    p.add_argument("--soft_clamp_limit", type=float, default=2.0 ** 12,
                   help="Pre-overflow tripwire threshold (on the SCALED "
                        "stream). Should never fire once shrinks are sized "
                        "right; every CLAMP line tells you how many bits "
                        "short you are. Non-finite hits are reported as "
                        "poisoning (a saturating projection), which no "
                        "stream shrink fixes.")
    p.add_argument("--fp16_probe", action="store_true",
                   help="Install per-(step,block) absmax/p99.9/headroom "
                        "telemetry; prints a summary sorted worst-first and "
                        "writes fp16_stats.csv to --output_dir at exit. For "
                        "clean numbers combine with --no_sanitize.")
    p.add_argument("--fp16_trace_nonfinite", action="store_true",
                   help="Hook every leaf submodule of every DiT block and "
                        "report where NaN/Inf FIRST appears, distinguishing "
                        "'input already poisoned (upstream)' from 'overflow "
                        "originates here'. Slow; debug runs only.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    generate(_parse_args())