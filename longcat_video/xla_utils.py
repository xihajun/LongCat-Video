"""Utilities for running LongCat-Video on TPU via PyTorch/XLA GSPMD.

Design notes
------------
* A single 1D device mesh ("model") is registered globally. Linear weights are
  sharded Megatron-style (column-parallel for qkv/w1/w3/adaLN, row-parallel for
  proj/w2); GSPMD inserts the required collectives automatically.
* Full self-attention over ~37k tokens cannot use F.scaled_dot_product_attention
  on XLA: it decomposes into math attention and materializes the full
  [B, H, S, S] score matrix (tens of GB). `spmd_self_attention` dispatches to
  the best available implementation:
    1. Pallas FlashAttention (`torch_xla.experimental.custom_kernel`): one
       fused TPU kernel, O(S) memory, small HLO -> fast compiles and small
       compiled programs. Heads are sharded over the 'model' mesh axis via the
       kernel's manual-sharding support.
    2. `spmd_chunked_attention`: query-chunked matmul attention, used when the
       kernel is unavailable (older torch_xla) or LONGCAT_XLA_FLASH=0.
  Cross-attention (text kv<=512, audio kv~32) is tiny and can use the regular
  SDPA fallback.
* Everything degrades gracefully: if no mesh is registered or the tensors are
  not on an XLA device, these helpers are no-ops / plain PyTorch.
"""

import os
from typing import Optional

import torch

_GLOBAL_MESH = None


def set_global_mesh(mesh) -> None:
    global _GLOBAL_MESH
    _GLOBAL_MESH = mesh


def get_global_mesh():
    return _GLOBAL_MESH


def is_xla_tensor(t: torch.Tensor) -> bool:
    return t.device.type == "xla"


def maybe_mark_sharding(t: torch.Tensor, partition_spec) -> torch.Tensor:
    """Annotate `t` with a partition spec if SPMD is active. Never raises."""
    mesh = _GLOBAL_MESH
    if mesh is None or not is_xla_tensor(t):
        return t
    try:
        import torch_xla.distributed.spmd as xs
        xs.mark_sharding(t, mesh, partition_spec)
    except Exception:
        pass
    return t


# --------------------------------------------------------------------------
# Pallas FlashAttention
# --------------------------------------------------------------------------

_FLASH_KERNEL = None  # None = not probed yet, False = unavailable, else the kernel
_FLASH_BLOCK = 512    # kernel block size; seq lens are padded to a multiple of it


def _get_flash_kernel():
    """Probe once for torch_xla's Pallas flash-attention kernel."""
    global _FLASH_KERNEL
    if _FLASH_KERNEL is None:
        if os.environ.get("LONGCAT_XLA_FLASH", "1") == "0":
            _FLASH_KERNEL = False
            print("[xla-attn] Pallas flash attention disabled (LONGCAT_XLA_FLASH=0); "
                  "using chunked attention", flush=True)
        else:
            try:
                from torch_xla.experimental.custom_kernel import flash_attention
                _FLASH_KERNEL = flash_attention
                print("[xla-attn] Pallas flash attention kernel enabled", flush=True)
            except ImportError as e:
                _FLASH_KERNEL = False
                print(f"[xla-attn] Pallas flash attention unavailable ({e}); "
                      "using chunked attention", flush=True)
    return _FLASH_KERNEL


def _pad_seq(t: torch.Tensor, multiple: int):
    """Zero-pad dim 2 (sequence) of [B, H, S, D] up to a multiple. -> (t, pad)."""
    pad = (-t.shape[2]) % multiple
    if pad:
        t = torch.nn.functional.pad(t, (0, 0, 0, pad))
    return t, pad


def spmd_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> Optional[torch.Tensor]:
    """Full attention as a single Pallas TPU kernel. q, k, v: [B, H, S, D].

    The kernel requires sequence lengths to be multiples of its block size, so
    q/k/v are zero-padded and the padding is masked INSIDE the kernel with
    segment ids (a zero-padded key would otherwise still receive softmax
    weight: its score is 0, not -inf). Heads are sharded over the 'model' mesh
    axis through the kernel's manual-sharding integration.

    Returns [B, H, Sq, D] in q.dtype, or None if the kernel is unavailable
    (callers fall back to `spmd_chunked_attention`).
    """
    flash = _get_flash_kernel()
    if flash is False:
        return None

    sq, skv = q.shape[2], k.shape[2]
    q, pad_q = _pad_seq(q, _FLASH_BLOCK)
    k, pad_kv = _pad_seq(k, _FLASH_BLOCK)
    v, _ = _pad_seq(v, _FLASH_BLOCK)

    q_seg = kv_seg = None
    if pad_q or pad_kv:
        b = q.shape[0]
        q_seg = torch.zeros(b, q.shape[2], dtype=torch.float32, device=q.device)
        q_seg[:, :sq] = 1.0
        kv_seg = torch.zeros(b, k.shape[2], dtype=torch.float32, device=k.device)
        kv_seg[:, :skv] = 1.0

    kwargs = {}
    if _GLOBAL_MESH is not None:
        kwargs = dict(partition_spec=(None, "model", None, None), mesh=_GLOBAL_MESH)

    try:
        out = flash(q, k, v, False, q_seg, kv_seg, scale, **kwargs)
    except Exception as e:
        global _FLASH_KERNEL
        _FLASH_KERNEL = False
        print(f"[xla-attn] flash attention kernel failed ({e}); "
              "falling back to chunked attention", flush=True)
        return None

    return out[:, :, :sq] if pad_q else out


def spmd_self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Self-attention for XLA: Pallas flash kernel if available, else chunked."""
    out = spmd_flash_attention(q, k, v, scale)
    if out is not None:
        return out
    return spmd_chunked_attention(q, k, v, scale, chunk_size)


# --------------------------------------------------------------------------
# Chunked fallback
# --------------------------------------------------------------------------

def spmd_chunked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Memory-bounded full attention for XLA. q, k, v: [B, H, S, D].

    Computes softmax(q k^T * scale) v in query chunks. Scores use fp32 softmax
    for numerical parity with flash attention. The head dimension (dim 1) is
    annotated with the 'model' mesh axis so each TPU core keeps only its heads.
    Returns [B, H, Sq, D] in q.dtype.
    """
    for t in (q, k, v):
        maybe_mark_sharding(t, (None, "model", None, None))

    kt = k.transpose(-1, -2)
    outputs = []
    for qc in torch.split(q, chunk_size, dim=2):
        scores = torch.matmul(qc, kt) * scale            # [B, H, c, Skv]
        probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        outputs.append(torch.matmul(probs, v))           # [B, H, c, D]
    out = torch.cat(outputs, dim=2)
    maybe_mark_sharding(out, (None, "model", None, None))
    return out
