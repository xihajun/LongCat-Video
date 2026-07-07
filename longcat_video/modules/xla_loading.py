"""Load the LongCat-Video-Avatar DiT onto a TPU slice with PyTorch/XLA GSPMD.

Strategy (v5e-8: 8 chips x 16 GB HBM = 128 GB):
  1. Build the model skeleton on the `meta` device (no memory).
  2. Stream the bf16 safetensors shards into CPU tensors, assigning
     parameter-by-parameter (peak host RAM ~= model size + one shard).
  3. Merge the DMD distillation LoRA into the base weights on CPU (fp32 math),
     so the TPU graph contains no LoRA side-branches and nothing is replicated.
  4. Move every parameter to the XLA virtual device and annotate it with a
     Megatron-style 1D sharding over the 'model' mesh axis:
        - column-parallel (shard dim 0): qkv, q_linear, kv_linear, w1, w3,
          adaLN modulations, big MLPs — plus their biases
        - row-parallel   (shard dim 1): attention proj, ffn w2
        - replicated: norms, small embedders, final_layer, anything whose
          shard dim is not divisible by the mesh size
     GSPMD propagates these through the graph and inserts collectives.

bf16 weights: ~31.7 GB total -> ~4 GB per chip. No quantization needed.
"""

import json
import os
import re
from typing import Optional

import torch

# Column-parallel (shard output dim / dim 0) if the name matches; the bias of a
# column-parallel linear is sharded on dim 0 as well.
_COL_PATTERNS = (
    r"\.qkv\.(weight|bias)$",
    r"\.q_linear\.(weight|bias)$",
    r"\.kv_linear\.(weight|bias)$",
    r"\.w1\.weight$",
    r"\.w3\.weight$",
    r"adaLN_modulation\.\d+\.(weight|bias)$",
    r"audio_adaln\w*\.\d+\.(weight|bias)$",
)
# Row-parallel (shard input dim / dim 1); bias replicated.
_ROW_PATTERNS = (
    r"\.attn\.proj\.weight$",
    r"\.cross_attn\.proj\.weight$",
    r"\.audio_attn\w*\.proj\.weight$",
    r"blocks\.\d+\..*\bproj\.weight$",
    r"\.w2\.weight$",
)


def _partition_spec_for(name: str, tensor: torch.Tensor, mesh_size: int):
    """Return a partition spec tuple or None (replicated)."""
    if tensor.ndim == 0:
        return None
    for pat in _ROW_PATTERNS:
        if re.search(pat, name):
            if tensor.ndim == 2 and tensor.shape[1] % mesh_size == 0:
                return (None, "model")
            return None
    for pat in _COL_PATTERNS:
        if re.search(pat, name):
            if tensor.shape[0] % mesh_size == 0:
                if tensor.ndim == 2:
                    return ("model", None)
                if tensor.ndim == 1:
                    return ("model",)
            return None
    return None


def merge_lora_into_state_(model: torch.nn.Module, lora_path: str,
                           multiplier: float = 1.0, network_dim: int = 128,
                           network_alpha: float = 64.0) -> int:
    """Merge a LongCat LoRA checkpoint into the (CPU-resident) base weights.

    Key format (see lora_utils.LoRANetwork):
      lora___lorahyphen___<module___lorahyphen___path>.lora_down.weight
      ...lora_up.weight                (n_seperate == 1)
      ...lora_up.blocks.<i>.weight     (n_seperate  > 1; output rows partitioned)

    delta_W = multiplier * (alpha / dim) * up @ down, computed in fp32.
    Returns the number of modules merged.
    """
    from safetensors.torch import load_file

    sd = load_file(lora_path, device="cpu")
    scale = multiplier * (float(network_alpha) / float(network_dim))
    hy = "___lorahyphen___"

    module_names = sorted({k.split(".lora_down.weight")[0]
                           for k in sd if k.endswith(".lora_down.weight")})
    merged = 0
    for lora_name in module_names:
        module_path = lora_name.replace("lora" + hy, "").replace(hy, ".")
        try:
            mod = model
            for part in module_path.split("."):
                mod = getattr(mod, part)
            weight = mod.weight
        except AttributeError:
            print(f"[lora-merge] module not found, skipped: {module_path}")
            continue

        down = sd[f"{lora_name}.lora_down.weight"].float()           # [n*r, in]
        up_key = f"{lora_name}.lora_up.weight"
        if up_key in sd:
            delta = sd[up_key].float() @ down                        # [out, in]
        else:
            blk_keys = sorted(
                (k for k in sd if k.startswith(f"{lora_name}.lora_up.blocks.")),
                key=lambda k: int(k.split(".blocks.")[1].split(".")[0]))
            r = network_dim
            parts = [sd[k].float() @ down[i * r:(i + 1) * r, :]
                     for i, k in enumerate(blk_keys)]
            delta = torch.cat(parts, dim=0)

        assert delta.shape == weight.shape, \
            f"{module_path}: lora delta {tuple(delta.shape)} vs weight {tuple(weight.shape)}"
        with torch.no_grad():
            weight.copy_((weight.float() + scale * delta).to(weight.dtype))
        merged += 1
    print(f"[lora-merge] merged {merged} modules (scale={scale})")
    return merged


def load_dit_xla_spmd(
    checkpoint_dir: str,
    mesh,
    subfolder: str = "base_model",
    dtype: torch.dtype = torch.bfloat16,
    lora_path: Optional[str] = None,
    lora_multiplier: float = 1.0,
    lora_dim: int = 128,
    lora_alpha: float = 64.0,
    **kwargs,
):
    """Build the bf16 avatar DiT and shard it across the TPU mesh. Returns the model."""
    import torch_xla
    import torch_xla.distributed.spmd as xs
    from safetensors import safe_open
    from .avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
    from ..xla_utils import set_global_mesh

    set_global_mesh(mesh)
    mesh_size = int(len(getattr(mesh, "device_ids", [])) or 1)

    model_dir = os.path.join(checkpoint_dir, subfolder)
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    for k in ("_class_name", "architectures", "_diffusers_version", "model_max_length"):
        config.pop(k, None)
    config.update(kwargs)
    # No flash-attn / xformers on TPU; the XLA chunked-attention path is used.
    for flag in ("enable_flashattn3", "enable_flashattn2", "enable_xformers"):
        if flag not in kwargs:
            config[flag] = False

    # 1) meta skeleton
    with torch.device("meta"):
        model = LongCatVideoAvatarTransformer3DModel(**config)
    model.eval().requires_grad_(False)

    # 2) stream shards into CPU tensors
    shard_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    assert shard_files, f"no .safetensors found in {model_dir}"
    modules = dict(model.named_modules())
    loaded = 0
    for fname in shard_files:
        with safe_open(os.path.join(model_dir, fname), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype)
                mod_path, _, leaf = key.rpartition(".")
                sub = modules.get(mod_path)
                if sub is None:
                    print(f"[xla-load] unexpected key skipped: {key}")
                    continue
                if leaf in sub._parameters and sub._parameters[leaf] is not None:
                    sub._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)
                elif leaf in sub._buffers:
                    sub._buffers[leaf] = tensor
                else:
                    print(f"[xla-load] key has no slot, skipped: {key}")
                    continue
                loaded += 1
    print(f"[xla-load] assigned {loaded} tensors from {len(shard_files)} shard(s)")

    metas = [n for n, p in model.named_parameters() if p.device.type == "meta"] + \
            [n for n, b in model.named_buffers() if b.device.type == "meta"]
    assert not metas, f"tensors still on meta after load: {metas[:8]}"

    # 3) merge the distillation LoRA on CPU
    if lora_path is not None:
        merge_lora_into_state_(model, lora_path, lora_multiplier, lora_dim, lora_alpha)

    # 4) transfer to the XLA virtual device with sharding annotations
    sharded, replicated = 0, 0
    for name, param in model.named_parameters():
        spec = _partition_spec_for(name, param, mesh_size)
        xla_param = torch.nn.Parameter(param.data.to("xla"), requires_grad=False)
        if spec is not None:
            xs.mark_sharding(xla_param, mesh, spec)
            sharded += 1
        else:
            replicated += 1
        mod_path, _, leaf = name.rpartition(".")
        modules[mod_path]._parameters[leaf] = xla_param
    for name, buf in model.named_buffers():
        mod_path, _, leaf = name.rpartition(".")
        modules[mod_path]._buffers[leaf] = buf.to("xla")
    torch_xla.sync()
    print(f"[xla-load] parameters sharded: {sharded}, replicated: {replicated}")
    return model
