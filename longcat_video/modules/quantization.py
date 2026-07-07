import os
import json
import shutil
from typing import Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file


class QuantizedLinear(nn.Module):
    """INT8 weight-only quantized linear layer with per-channel symmetric quantization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_int8", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.zeros(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weight to input dtype for computation
        compute_dtype = x.dtype
        weight = self.weight_int8.to(compute_dtype) * self.weight_scale.to(compute_dtype).unsqueeze(1)
        bias = self.bias.to(compute_dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QuantizedLinear":
        """Quantize a standard Linear layer to INT8."""
        has_bias = linear.bias is not None
        ql = cls(linear.in_features, linear.out_features, bias=has_bias)

        weight = linear.weight.data.float()
        # Per-channel symmetric quantization
        scale = weight.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        weight_int8 = (weight / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

        ql.weight_int8 = weight_int8
        ql.weight_scale = scale
        if has_bias:
            ql.bias = linear.bias.data.to(torch.bfloat16)
        return ql

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


# Layers to skip quantization (sensitive to precision)
DEFAULT_SKIP_PATTERNS = {
    "final_layer.linear",  # Final output projection, precision-sensitive
}


def quantize_model(model: nn.Module, skip_patterns: Optional[Set[str]] = None) -> nn.Module:
    """Replace all nn.Linear layers in the model with QuantizedLinear (INT8 weight-only).

    Args:
        model: The model to quantize (modified in-place).
        skip_patterns: Set of module name patterns to skip. If None, uses DEFAULT_SKIP_PATTERNS.

    Returns:
        The quantized model.
    """
    if skip_patterns is None:
        skip_patterns = DEFAULT_SKIP_PATTERNS

    modules_to_replace = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module should be skipped
            should_skip = any(pattern in name for pattern in skip_patterns)
            if not should_skip:
                modules_to_replace[name] = module

    for name, linear in modules_to_replace.items():
        # Navigate to the parent module
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        # Replace with quantized version
        setattr(parent, parts[-1], QuantizedLinear.from_linear(linear))

    return model


def save_quantized_state_dict(model: nn.Module, save_dir: str, config_source_dir: Optional[str] = None):
    """Save the quantized model's state dict using safetensors format.

    Saves quantized weights in shards to avoid single large files.

    Args:
        model: The quantized model.
        save_dir: Directory to save quantized weights.
        config_source_dir: If provided, copy config.json from this directory.
    """
    os.makedirs(save_dir, exist_ok=True)

    state_dict = model.state_dict()

    # Split into shards (~4GB each for manageable files)
    max_shard_size = 4 * 1024 * 1024 * 1024  # 4GB

    shards = []
    current_shard = {}
    current_size = 0

    for key, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > max_shard_size and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    # Save each shard
    index = {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in state_dict.values())}, "weight_map": {}}

    for i, shard in enumerate(shards):
        shard_name = f"quantized_model-{i+1:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, os.path.join(save_dir, shard_name))
        for key in shard:
            index["weight_map"][key] = shard_name

    # Save index
    index_path = os.path.join(save_dir, "quantized_model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    # Copy config if source provided
    if config_source_dir:
        src_config = os.path.join(config_source_dir, "config.json")
        if os.path.exists(src_config):
            shutil.copy2(src_config, os.path.join(save_dir, "config.json"))

    # Save quantization metadata
    quant_config = {
        "quantization_method": "int8_per_channel_symmetric",
        "skip_patterns": list(DEFAULT_SKIP_PATTERNS),
        "description": "Weight-only INT8 quantization with per-channel symmetric scaling"
    }
    with open(os.path.join(save_dir, "quantization_config.json"), "w") as f:
        json.dump(quant_config, f, indent=2)

    print(f"Saved {len(shards)} shard(s) to {save_dir}")


def load_quantized_dit(checkpoint_dir: str, subfolder: str = "base_model_int8", **kwargs):
    """Load a quantized DiT model.

    Args:
        checkpoint_dir: Base checkpoint directory.
        subfolder: Subfolder containing quantized weights (default: 'base_model_int8').
        **kwargs: Additional kwargs passed to the model constructor (e.g., cp_split_hw).

    Returns:
        The quantized DiT model ready for inference.
    """
    from .avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel

    quantized_dir = os.path.join(checkpoint_dir, subfolder)

    # Load config
    config_path = os.path.join(quantized_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Remove non-constructor keys
    config.pop("_class_name", None)
    config.pop("architectures", None)
    config.pop("_diffusers_version", None)
    config.pop("model_max_length", None)

    # Override with kwargs
    config.update(kwargs)

    # Instantiate model (empty weights)
    model = LongCatVideoAvatarTransformer3DModel(**config)

    # Replace Linear layers with QuantizedLinear (empty)
    skip_patterns = DEFAULT_SKIP_PATTERNS
    modules_to_replace = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            should_skip = any(pattern in name for pattern in skip_patterns)
            if not should_skip:
                ql = QuantizedLinear(module.in_features, module.out_features, bias=module.bias is not None)
                modules_to_replace[name] = ql

    for name, ql in modules_to_replace.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], ql)

    # Load quantized state dict
    index_path = os.path.join(quantized_dir, "quantized_model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        # Load from shards
        shard_files = set(index["weight_map"].values())
        state_dict = {}
        for shard_file in sorted(shard_files):
            shard_path = os.path.join(quantized_dir, shard_file)
            shard_dict = load_file(shard_path, device="cpu")
            state_dict.update(shard_dict)
    else:
        # Single file fallback
        files = [f for f in os.listdir(quantized_dir) if f.endswith(".safetensors") and "index" not in f]
        state_dict = {}
        for f in sorted(files):
            shard_dict = load_file(os.path.join(quantized_dir, f), device="cpu")
            state_dict.update(shard_dict)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Cast non-quantized parameters (Conv3d, LayerNorm, etc.) to bfloat16
    # QuantizedLinear buffers (int8, float32 scale) are kept as-is
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            continue
        for param_name, param in module.named_parameters(recurse=False):
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)

    return model


def load_quantized_dit_sharded(
    checkpoint_dir: str,
    subfolder: str = "base_model_int8",
    devices=("cuda:0", "cuda:1"),
    split_index: Optional[int] = None,
    compute_dtype: torch.dtype = torch.float16,
    **kwargs,
):
    """Low-CPU-memory loader that streams the INT8 checkpoint tensor-by-tensor
    straight onto two GPUs (model sharding / pipeline parallelism).

    Why this exists: the stock ``load_quantized_dit`` (a) instantiates the full
    model on CPU (~16 GB of real zeros for the INT8 buffers) and then (b) loads
    the entire state dict into CPU RAM (~16 GB more) before ``load_state_dict``
    — a ~32 GB CPU peak that gets SIGKILL'ed on a 30 GB Kaggle box. Here the
    skeleton is built on the ``meta`` device (no memory), and every tensor is
    copied from the memory-mapped safetensors shard directly to its target GPU.
    Peak CPU usage stays well under 2 GB.

    Args:
        checkpoint_dir: e.g. ``.../LongCat-Video-Avatar-1.5``
        subfolder:      ``base_model_int8``
        devices:        two CUDA devices; blocks[:split_index] + embedders go to
                        devices[0], blocks[split_index:] + final_layer to devices[1]
        split_index:    block index where the pipeline stage boundary sits
                        (default: depth//2 rounded up)
        compute_dtype:  dtype for non-INT8 floating tensors. T4 (sm75) has no
                        bf16 tensor cores, so float16 is strongly recommended.
    """
    from safetensors import safe_open
    from .avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel

    quantized_dir = os.path.join(checkpoint_dir, subfolder)

    with open(os.path.join(quantized_dir, "config.json"), "r") as f:
        config = json.load(f)
    for k in ("_class_name", "architectures", "_diffusers_version", "model_max_length"):
        config.pop(k, None)
    config.update(kwargs)
    # The T4 (sm75) has no flash-attn support (needs sm80+). Unless the caller
    # explicitly requests a backend via kwargs, force everything off so the
    # SDPA fallback is used.
    for flag in ("enable_flashattn3", "enable_flashattn2", "enable_xformers"):
        if flag not in kwargs:
            config[flag] = False

    # 1) Build the skeleton on the meta device: zero real memory allocated.
    with torch.device("meta"):
        model = LongCatVideoAvatarTransformer3DModel(**config)

        # Swap nn.Linear -> QuantizedLinear (still on meta) exactly like the
        # stock loader does, so state-dict keys line up.
        skip_patterns = DEFAULT_SKIP_PATTERNS
        modules_to_replace = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if not any(p in name for p in skip_patterns):
                    modules_to_replace[name] = QuantizedLinear(
                        module.in_features, module.out_features, bias=module.bias is not None
                    )
        for name, ql in modules_to_replace.items():
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], ql)

    depth = len(model.blocks)
    if split_index is None:
        split_index = (depth + 1) // 2
    split_index = max(1, min(depth - 1, int(split_index)))
    devices = [torch.device(d) for d in devices]

    def target_device(key: str) -> torch.device:
        if key.startswith("blocks."):
            block_idx = int(key.split(".")[1])
            return devices[0] if block_idx < split_index else devices[1]
        if key.startswith("final_layer"):
            return devices[1]
        return devices[0]  # x_embedder / t_embedder / y_embedder / audio_proj

    def convert_dtype(key: str, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype == torch.int8:
            return tensor
        if key.endswith("weight_scale"):
            return tensor.to(torch.float32)
        if tensor.is_floating_point():
            return tensor.to(compute_dtype)
        return tensor

    # 2) Discover shard files.
    index_path = os.path.join(quantized_dir, "quantized_model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = sorted(
            f for f in os.listdir(quantized_dir)
            if f.endswith(".safetensors") and "index" not in f
        )

    # 3) Stream tensors: mmap-read -> dtype convert -> copy to target GPU.
    loaded_keys = set()
    for shard_file in shard_files:
        shard_path = os.path.join(quantized_dir, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = convert_dtype(key, f.get_tensor(key))
                dev = target_device(key)
                parts = key.split(".")
                sub = model
                for part in parts[:-1]:
                    sub = getattr(sub, part)
                leaf = parts[-1]
                if leaf in sub._parameters and sub._parameters[leaf] is not None:
                    sub._parameters[leaf] = nn.Parameter(tensor.to(dev), requires_grad=False)
                elif leaf in sub._buffers:
                    sub._buffers[leaf] = tensor.to(dev)
                else:
                    print(f"[load_quantized_dit_sharded] unexpected key skipped: {key}")
                    continue
                loaded_keys.add(key)
                del tensor

    # 4) Sanity check: nothing may remain on meta.
    leftover = [n for n, p in model.named_parameters() if p.is_meta]
    leftover += [n for n, b in model.named_buffers() if b.is_meta]
    if leftover:
        raise RuntimeError(
            f"{len(leftover)} tensors were not found in the checkpoint and are still on "
            f"the meta device, e.g. {leftover[:5]}. The checkpoint layout does not match "
            f"the model definition."
        )

    model.eval()
    model.requires_grad_(False)
    model.shard_across_devices(devices=devices, split_index=split_index)

    n_dev0 = sum(
        (p.numel() * p.element_size()) for p in model.parameters() if p.device == devices[0]
    ) + sum((b.numel() * b.element_size()) for b in model.buffers() if b.device == devices[0])
    n_dev1 = sum(
        (p.numel() * p.element_size()) for p in model.parameters() if p.device == devices[1]
    ) + sum((b.numel() * b.element_size()) for b in model.buffers() if b.device == devices[1])
    print(
        f"[load_quantized_dit_sharded] blocks 0-{split_index-1} + embedders -> {devices[0]} "
        f"({n_dev0/1024**3:.2f} GB); blocks {split_index}-{depth-1} + final_layer -> {devices[1]} "
        f"({n_dev1/1024**3:.2f} GB)"
    )
    return model
