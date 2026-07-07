"""fp16_range_toolkit.py

Quantify and fix fp16 dynamic-range problems when running a bf16-trained
pre-norm DiT (e.g. LongCat-Video-Avatar) in float16.

Components:

1. install_block_stats(model)       -- per-(step, block) headroom telemetry
2. rescale_residual_stream(model)   -- exact static rescaling of the residual
                                       stream (generic version; the avatar
                                       model uses the model-specific
                                       rescale_residual_stream_avatar in
                                       run_avatar_t4_pp.py instead, which also
                                       folds branch shrinks into projection
                                       weights -- measured Inf originates
                                       INSIDE branch matmuls, which a stream
                                       rescale cannot reach)
3. install_soft_clamp(model)        -- pre-overflow clamp tripwire that logs
                                       (replacement for the post-Inf sanitizer)
4. install_nonfinite_tracer(model)  -- origin finder: which submodule FIRST
                                       produces NaN/Inf, and whether its input
                                       was already poisoned

All statistics are computed in fp32 on-GPU; overhead is one elementwise
reduction per block per forward, negligible next to the matmuls (the tracer
is much heavier -- debug runs only).
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict

import torch
import torch.nn as nn

FP16_MAX = 65504.0


def _first_tensor(out):
    """Extract the hidden-states tensor from a module output."""
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        for o in out:
            t = _first_tensor(o)
            if t is not None:
                return t
    if isinstance(out, dict):
        for o in out.values():
            t = _first_tensor(o)
            if t is not None:
                return t
    return None


# ---------------------------------------------------------------------------
# 0. Anatomy helper: know what you are hooking before you hook it.
# ---------------------------------------------------------------------------

def print_block_anatomy(model, block_index: int = 0):
    """Print submodules of one block and of final_layer.

    Use this to (a) confirm the entry norms are LayerNorm / RMSNorm
    (scale-invariant -> the rescale below is exact) and (b) find the real
    names of the residual-writer projections for `writer_patterns`.
    """
    blk = model.blocks[block_index]
    print(f"=== blocks[{block_index}] ===")
    for name, mod in blk.named_modules():
        if name:
            print(f"  {name:50s} {mod.__class__.__name__}")
    fl = getattr(model, "final_layer", None)
    if fl is not None:
        print("=== final_layer ===")
        for name, mod in fl.named_modules():
            if name:
                print(f"  {name:50s} {mod.__class__.__name__}")


# ---------------------------------------------------------------------------
# 1. Telemetry: per-(step, block) absmax / p99.9 / headroom-in-bits.
# ---------------------------------------------------------------------------

class BlockStats:
    def __init__(self):
        # rows: (step, block_name, absmax, p999, headroom_bits)
        self.rows = []
        self._step = 0
        self._handles = []
        self._sample = 1 << 20  # elements sampled for the quantile

    # -- hook bodies --------------------------------------------------------

    def _on_root_pre(self, module, args, kwargs):
        self._step += 1
        return None

    def _make_hook(self, name):
        def hook(module, inputs, output):
            t = _first_tensor(output)
            if t is None or not t.is_floating_point():
                return
            x = t.detach()
            flat = x.reshape(-1)
            absmax = flat.abs().amax().float()
            n = flat.numel()
            if n > self._sample:
                idx = torch.randint(n, (self._sample,), device=flat.device)
                sample = flat[idx].float().abs()
            else:
                sample = flat.float().abs()
            sample = torch.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)
            p999 = torch.quantile(sample, 0.999)
            am = absmax.item()
            headroom = math.log2(FP16_MAX / am) if am > 0 and math.isfinite(am) else float("-inf") if not math.isfinite(am) else float("inf")
            self.rows.append((self._step, name, am, p999.item(), headroom))
        return hook

    # -- public API ----------------------------------------------------------

    def install(self, model, include_final_layer: bool = True):
        h = model.register_forward_pre_hook(self._on_root_pre, with_kwargs=True)
        self._handles.append(h)
        for i, blk in enumerate(model.blocks):
            self._handles.append(
                blk.register_forward_hook(self._make_hook(f"blocks[{i}]"))
            )
        fl = getattr(model, "final_layer", None)
        if include_final_layer and fl is not None:
            self._handles.append(
                fl.register_forward_hook(self._make_hook("final_layer"))
            )
        return self

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def summary(self):
        worst = {}
        for step, name, am, p999, hr in self.rows:
            cur = worst.get(name)
            if cur is None or am > cur[0]:
                worst[name] = (am, p999, hr, step)
        print(f"{'block':14s} {'absmax':>12s} {'p99.9':>12s} "
              f"{'headroom(bits)':>15s} {'@step':>6s}  flag")
        for name in sorted(worst, key=lambda k: worst[k][2]):
            am, p999, hr, step = worst[name]
            flag = ("OVERFLOW" if hr < 0
                    else "DANGER" if hr < 2
                    else "ok")
            outlier = (am / p999) if p999 > 0 else float("inf")
            note = f" outlier-ratio={outlier:.0f}x" if outlier > 100 else ""
            print(f"{name:14s} {am:12.4g} {p999:12.4g} {hr:15.2f} "
                  f"{step:6d}  {flag}{note}")
        return worst

    def suggest_shrink(self, target_bits: float = 2.0) -> float:
        """Power-of-two shrink factor that restores `target_bits` of headroom
        at the globally worst observed absmax. Returns 1.0 if none needed.

        NOTE: if any absmax was Inf/NaN this cannot be sized from data --
        that is a POISONING problem, not a headroom problem; locate the
        source with install_nonfinite_tracer first."""
        finite = [r[4] for r in self.rows if math.isfinite(r[4])]
        if not finite:
            return 1.0
        deficit = target_bits - min(finite)
        if deficit <= 0:
            return 1.0
        return float(2 ** math.ceil(deficit))

    def to_csv(self, path: str):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "block", "absmax", "p999", "headroom_bits"])
            w.writerows(self.rows)
        print(f"[fp16-toolkit] wrote {len(self.rows)} rows -> {path}")


def install_block_stats(model, **kw) -> BlockStats:
    return BlockStats().install(model, **kw)


# ---------------------------------------------------------------------------
# 2. Generic fix: exact static rescaling of the residual stream.
#
# Pre-norm block:   x_{l+1} = x_l + gate * Branch(Norm(x_l))
# With x -> x/s and every residual-writer output scaled by 1/s, the whole
# stream is uniformly x/s. Since LayerNorm/RMSNorm are scale-invariant, every
# Branch(...) and final_layer(Norm(...)) sees identical inputs -> the model
# output is mathematically unchanged, while the fp16 stream gains log2(s)
# bits of headroom.
#
# CAVEATS measured on LongCat-Avatar: (1) the audio branch re-norms its
# writer output before the gated residual add, so writer hooks there are
# cancelled -- scale the adaLN gates instead; (2) fp16 saturation can
# originate INSIDE branch projections (inputs come from LayerNorms and are
# invisible to any stream rescale) -- fold shrinks into projection weights
# and compensate in the gates. Both are done by the model-specific
# rescale_residual_stream_avatar() in run_avatar_t4_pp.py.
# ---------------------------------------------------------------------------

DEFAULT_WRITER_PATTERNS = (
    "to_out", "o_proj", "proj_out", "out_proj",   # attention output proj
    "ffn.w2", "ffn.fc2", "mlp.fc2", "down_proj",  # FFN down projection
)

DEFAULT_SOURCE_PATTERNS = (
    "patch_embed", "x_embedder", "proj_in",
)


class _Rescale:
    def __init__(self, s):
        self.inv = 1.0 / float(s)

    def __call__(self, module, inputs, output):
        t = _first_tensor(output)
        if t is None:
            return output
        if torch.is_tensor(output):
            return output * self.inv
        if isinstance(output, tuple):
            return tuple(
                (o * self.inv if torch.is_tensor(o) and o is t else o)
                for o in output
            )
        return output


def rescale_residual_stream(
    model,
    shrink: float = 16.0,
    writer_patterns=DEFAULT_WRITER_PATTERNS,
    source_patterns=DEFAULT_SOURCE_PATTERNS,
    dry_run: bool = False,
):
    """Scale the residual stream by 1/shrink, exactly (up to fp rounding).

    Requirements (verify with print_block_anatomy):
      * every block reads x only through a scale-invariant Norm;
      * final_layer also norms its input first (standard DiT);
      * writer_patterns covers EVERY module whose output is ADDED to the
        stream, source_patterns covers every module whose output BECOMES
        the stream, and no norm sits between a writer and the add.

    Returns list of hook handles (call .remove() on each to undo).
    """
    assert shrink >= 1.0
    if shrink == 1.0:
        print("[fp16-toolkit] shrink=1 -> nothing to do")
        return []

    targets = []
    for name, mod in model.named_modules():
        if any(p in name for p in writer_patterns) or \
           any(p in name for p in source_patterns):
            if len(list(mod.children())) == 0 or isinstance(mod, nn.Linear):
                targets.append((name, mod))

    if not targets:
        raise RuntimeError(
            "rescale_residual_stream matched no modules -- run "
            "print_block_anatomy(model) and fix writer/source patterns."
        )

    print(f"[fp16-toolkit] residual rescale 1/{shrink:g} on "
          f"{len(targets)} modules:")
    for name, _ in targets:
        print(f"    {name}")
    if dry_run:
        print("[fp16-toolkit] dry_run=True -> no hooks installed")
        return []

    hooks = [mod.register_forward_hook(_Rescale(shrink))
             for _, mod in targets]
    print(f"[fp16-toolkit] installed. Stream now carries x/{shrink:g} "
          f"(+{math.log2(shrink):.0f} bits headroom); output is unchanged "
          f"because entry norms are scale-invariant.")
    return hooks


# ---------------------------------------------------------------------------
# 3. Tripwire: proactive soft clamp with counters.
#
# NOT a fix -- a safety net that should never fire once the rescale is sized
# correctly. Unlike post-Inf nan_to_num it (a) acts BEFORE saturation,
# (b) preserves sign/ordering instead of zeroing, (c) tells you exactly how
# many bits short you are. A NON-FINITE hit is reported separately: that is
# a branch projection saturating fp16 internally -- fold factors, not stream
# shrink, are the lever.
# ---------------------------------------------------------------------------

def install_soft_clamp(model, limit: float = 2.0 ** 12, log_fn=print,
                       magnitude_clamp: bool = True):
    """Per-block sentinel.

    magnitude_clamp=True  (fp16-stream mode): clamp |x| to `limit` AND scrub
        NaN/Inf, replacing them with ±limit. IMPORTANT: keep limit <= 2^13.
        The v2 default of 2^15 = 32768 was a design bug: nan_to_num wrote
        ±32768 spikes into the stream, and the very next residual add of two
        such values gave 65536 > 65504 -> a NEW Inf. That self-sustaining
        cascade is how ~9k bad values in blocks[0] became 100% corruption
        (150,994,944 = every element of a 36864-token x 4096-dim tensor) by
        step 5. At 4096, even worst-case compounding stays far from 65504.

    magnitude_clamp=False (fp32-stream mode): the stream may LEGALLY exceed
        65504 in fp32, so never clamp magnitude; only scrub true NaN/Inf
        (replaced with 0) and report loudly -- with the fp32 stream this
        should never fire.
    """
    counts = defaultdict(int)

    def make(name):
        def hook(module, inputs, output):
            t = _first_tensor(output)
            if t is None:
                return output
            am = t.abs().amax().item()
            if not math.isfinite(am):
                counts[name] += 1
                n_bad = (~torch.isfinite(t)).sum().item()
                log_fn(f"[fp16-toolkit] CLAMP {name}: NON-FINITE output "
                       f"({n_bad} bad values, hit #{counts[name]}) -- a "
                       f"branch matmul is saturating fp16 internally; raise "
                       f"its fold factor (attn_shrink / ffn_shrink / "
                       f"ffn_hidden_shrink / cross_shrink) or rerun the "
                       f"tracer to locate it")
                repl = limit if magnitude_clamp else 0.0
                torch.nan_to_num_(t, nan=0.0, posinf=repl, neginf=-repl)
                if magnitude_clamp:
                    t.clamp_(-limit, limit)
                return output
            if magnitude_clamp and am > limit:
                counts[name] += 1
                over_bits = math.log2(am / limit)
                log_fn(f"[fp16-toolkit] CLAMP {name}: absmax={am:.3g} "
                       f"(+{over_bits:.2f} bits over limit, "
                       f"hit #{counts[name]}) -- increase shrink by "
                       f"2^{math.ceil(over_bits)}")
                t.clamp_(-limit, limit)
            return output
        return hook

    handles = [blk.register_forward_hook(make(f"blocks[{i}]"))
               for i, blk in enumerate(model.blocks)]
    mode = (f"magnitude+NaN/Inf at ±{limit:g}" if magnitude_clamp
            else "NaN/Inf sentinel only (fp32 stream)")
    print(f"[fp16-toolkit] soft clamp tripwire on {len(handles)} blocks "
          f"[{mode}] (expected hit count: 0)")
    return handles, counts


# ---------------------------------------------------------------------------
# 4. Origin finder: which submodule FIRST produces NaN/Inf?
#
# Hooks every leaf submodule of every block (plus x_embedder / final_layer).
# For each of the first `max_reports` non-finite outputs it reports whether
# the module's INPUT was already poisoned (upstream problem) or finite
# (the overflow ORIGINATES here). NOTE: a block-level hit with NO leaf-level
# hit means the Inf was created by an INLINE op (gate multiply, modulate,
# residual add) on finite-but-near-max operands. Heavy (one sync per leaf
# per forward) -- debug runs only.
# ---------------------------------------------------------------------------

def install_nonfinite_tracer(model, max_reports: int = 40, log_fn=print):
    state = {"n": 0}
    handles = []

    def make(name):
        def hook(module, inputs, output):
            if state["n"] >= max_reports:
                return
            t = _first_tensor(output)
            if t is None or not t.is_floating_point():
                return
            if bool(torch.isfinite(t).all()):
                return
            ins = [i for i in inputs
                   if torch.is_tensor(i) and i.is_floating_point()]
            in_bad = any(bool((~torch.isfinite(i)).any()) for i in ins)
            in_am = max(
                (torch.nan_to_num(i.detach().float(), nan=0.0, posinf=0.0,
                                  neginf=0.0).abs().max().item()
                 for i in ins),
                default=float("nan"),
            )
            n_bad = int((~torch.isfinite(t)).sum())
            state["n"] += 1
            verdict = ("input ALREADY non-finite -> poison came from UPSTREAM"
                       if in_bad else
                       f"input finite (absmax={in_am:.3g}) "
                       f"-> overflow ORIGINATES HERE")
            log_fn(f"[fp16-toolkit] NONFINITE #{state['n']} {name}: "
                   f"{n_bad} bad output values; {verdict}")
            if state["n"] >= max_reports:
                log_fn("[fp16-toolkit] tracer report limit reached "
                       "(further hits suppressed)")
        return hook

    targets = []
    m = getattr(model, "x_embedder", None)
    if m is not None:
        targets.append(("x_embedder", m))
    for i, blk in enumerate(model.blocks):
        for n, mod in blk.named_modules():
            if n and len(list(mod.children())) == 0:
                targets.append((f"blocks[{i}].{n}", mod))
        targets.append((f"blocks[{i}]", blk))
    fl = getattr(model, "final_layer", None)
    if fl is not None:
        targets.append(("final_layer", fl))

    for name, mod in targets:
        handles.append(mod.register_forward_hook(make(name)))
    print(f"[fp16-toolkit] non-finite tracer on {len(handles)} modules "
          f"(reports first {max_reports} hits; slow -- debug only)")
    return handles