"""Training utilities: mixed precision, gradient clipping, learned loss weights, phase timing and loss helpers."""
import math
import time
from collections import deque
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Loss helpers ---------------------------------------------------------------------------------------------------
def masked_mean(x, mask):
    """Mean of x over mask > 0."""
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def weighted_moments(x, weights):
    """(mean, std) of x under non-negative weights of its shape (fp32 scalars, no host sync)."""
    x = x.float()
    wsum = weights.sum().clamp(min=1e-8)
    mean = (x * weights).sum() / wsum
    return mean, ((x - mean).square() * weights).sum().div(wsum).sqrt()


def explained_variance(pred, target, weights):
    """1 - Var(target - pred) / Var(target) under the weights, as a float (host sync: metrics only); 1 = exact,
    0 = no better than the mean, nan when the targets do not vary."""
    _, err = weighted_moments(target.float() - pred.float(), weights)
    _, spread = weighted_moments(target, weights)
    spread = float(spread)
    return 1.0 - float(err) ** 2 / spread ** 2 if spread > 1e-6 else float("nan")


def lambda_returns(rewards, values, continues, lambda_=0.95, valid=None):
    """TD(lambda) returns; rewards / continues [N, L], values [N, L+1]; invalid steps return their own value."""
    returns = torch.zeros_like(rewards)
    bootstrap = values[:, -1]
    for i in reversed(range(rewards.shape[-1])):
        ret = rewards[:, i] + continues[:, i] * ((1 - lambda_) * values[:, i + 1] + lambda_ * bootstrap)
        if valid is not None:
            ret = torch.where(valid[:, i] > 0, ret, values[:, i])
        returns[:, i] = ret
        bootstrap = ret
    return returns


def reinforce_loss(cfg, advantages, log_probs, entropies, alive, forced_first=None):
    """DreamerV3's normalised REINFORCE actor loss -> (loss, info).

    advantages [N, H]: normalised advantages of the sampled actions (no gradient); log_probs [N, H]: their log-probs
    (with gradient); entropies [N, H]; alive [N, H]: weight of each dream step (predicted survival, 0 = not counted).
    forced_first = (first_row, pi): dreams first_row.. start with a forced action (branch groups) of probability pi.

    loss = -sum(w sg(A) log pi) / sum(alive) - entropy_scale x entropy, w = alive; a forced first action is weighted
    by action_dim x pi(a), so each branch group gives exactly action_dim x the expected policy gradient at its state
    (its group-mean baseline drops out).
    info: "weights" (sample weights used).
    """
    adv = advantages.detach()
    w = alive
    if forced_first is not None:
        first, pi = forced_first
        w = w.clone()
        w[first:, 0] = w[first:, 0] * cfg.action_dim * pi.detach()
    loss = -((w * adv * log_probs).sum() / alive.sum().clamp(min=1.0)
             + cfg.entropy_scale * masked_mean(entropies, alive))
    return loss, {"weights": w}


@torch.no_grad()
def actor_metrics(advantages, alive):
    """Floats for the console / CSV: mean |A| over living steps."""
    return {"adv_abs_mean": masked_mean(advantages.abs(), alive).item()}


@contextmanager
def frozen(*modules):
    """Uses the modules as fixed functions: no gradient to their parameters (inputs still get one)."""
    params = [p for m in modules for p in m.parameters()]
    flags = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, f in zip(params, flags):
            p.requires_grad_(f)


# --- Mixed precision ------------------------------------------------------------------------------------------------
def resolve_amp_dtype(device, mode):
    """'auto' -> bf16 / fp16 on CUDA, off on CPU; 'bf16', 'fp16' or 'off' force it."""
    if mode in (None, False, "off"):
        return None
    if mode == "auto":
        if device.type != "cuda":
            return None
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[mode]


def make_grad_scaler(device, enabled, init_scale=1024.0):
    """Loss scaler, active only for fp16."""
    return torch.amp.GradScaler(device.type, init_scale=init_scale, enabled=enabled)


# --- Gradient clipping ----------------------------------------------------------------------------------------------
class PercentileClipper:
    """Clips gradient norms at a running percentile of recent pre-clip norms (fixed threshold while warming up)."""

    def __init__(self, percentile=99.0, window=2000, warmup=500, initial=100.0):
        self.percentile = percentile
        self.warmup = warmup
        self.initial = initial
        self.norms = deque(maxlen=window)
        self.n_seen = 0
        self._updates = self._clipped = 0

    def threshold(self):
        """Current clipping threshold."""
        if self.n_seen < self.warmup:
            return self.initial
        return float(np.percentile(np.fromiter(self.norms, dtype=np.float64), self.percentile))

    def record(self, norm, threshold):
        """Adds a pre-clip norm to the window (non-finite norms are skipped)."""
        if not math.isfinite(norm):
            return
        self.norms.append(norm)
        self.n_seen += 1
        self._updates += 1
        self._clipped += int(norm > threshold)

    def stats(self):
        """-> (threshold, fraction of updates clipped since the last call)."""
        frac = self._clipped / max(self._updates, 1)
        self._updates = self._clipped = 0
        return self.threshold(), frac

    def state_dict(self):
        """Recent norms and counter (checkpointed)."""
        return {"norms": list(self.norms), "n_seen": self.n_seen}

    def load_state_dict(self, state):
        """Restores the recent norms and counter."""
        self.norms.extend(state.get("norms", []))
        self.n_seen = int(state.get("n_seen", len(self.norms)))


class DevicePercentileClipper:
    """PercentileClipper with its state on the device, so threshold() and record() never wait for the GPU: the same
    window of the last `window` finite pre-clip norms, the same warm-up and threshold (linear-interpolated
    percentile), threshold() returns a 0-d tensor. Same state_dict format as PercentileClipper."""

    def __init__(self, percentile=99.0, window=2000, warmup=500, initial=100.0, device="cpu"):
        self.percentile, self.window, self.warmup, self.initial = percentile, window, warmup, initial
        self.buf = torch.full((window,), float("nan"), device=device)          # ring of finite norms (NaN = empty)
        self.ptr = torch.zeros(1, dtype=torch.long, device=device)              # next slot
        self.seen = torch.zeros((), dtype=torch.long, device=device)            # finite norms recorded
        self._clipped = torch.zeros((), device=device)
        self._updates = torch.zeros((), device=device)

    @property
    def n_seen(self):
        return int(self.seen)

    def threshold(self):
        """Running percentile of the window (the fixed initial threshold while warming up)."""
        q = torch.nanquantile(self.buf, self.percentile / 100.0)
        return torch.where(self.seen < self.warmup, torch.full_like(q, self.initial), q)

    def record(self, norm, threshold):
        """Adds a pre-clip norm to the window (non-finite norms are skipped, as in PercentileClipper)."""
        n = torch.as_tensor(norm, device=self.buf.device).detach().float().reshape(1)
        fin = torch.isfinite(n)
        self.buf.scatter_(0, self.ptr, torch.where(fin, n, self.buf.gather(0, self.ptr)))
        self.ptr.copy_(torch.where(fin, (self.ptr + 1) % self.window, self.ptr))
        self.seen += fin.long().squeeze(0)
        self._updates += fin.float().squeeze(0)
        self._clipped += (fin & (n > threshold)).float().squeeze(0)

    def stats(self):
        """-> (threshold, fraction of updates clipped since the last call)."""
        frac = float(self._clipped) / max(float(self._updates), 1.0)
        self._clipped.zero_()
        self._updates.zero_()
        return float(self.threshold()), frac

    def state_dict(self):
        v = torch.roll(self.buf, -int(self.ptr))
        return {"norms": v[~torch.isnan(v)].tolist(), "n_seen": int(self.seen)}

    def load_state_dict(self, state):
        norms = list(state.get("norms", []))[-self.window:]
        self.buf.fill_(float("nan"))
        if norms:
            self.buf[:len(norms)] = torch.tensor(norms, dtype=torch.float32, device=self.buf.device)
        self.ptr.fill_(len(norms) % self.window)
        self.seen.fill_(int(state.get("n_seen", len(norms))))


# --- torch.compile --------------------------------------------------------------------------------------------------
_EAGER_ONLY = [0]                                                      # > 0: compiled modules run eagerly


@contextmanager
def eager_modules():
    """Compiled modules run their eager forward inside this block (bodies of hand-captured CUDA graphs, the dream
    audit, the gradient-share measurement). Checkpointed passes then use plain recomputation (no compiled backward,
    no selective checkpointing), so a graph built here can be backpropagated more than once."""
    _EAGER_ONLY[0] += 1
    try:
        yield
    finally:
        _EAGER_ONLY[0] -= 1


def eager_only():
    """True inside eager_modules() (for compiled callables that are not modules: encoder layers, decoder chunks, TSSM
    blocks)."""
    return _EAGER_ONLY[0] > 0


def compile_module(module, **kwargs):
    """Compiles a module in place (state_dict keys unchanged, like nn.Module.compile), except inside eager_modules().
    Compile it only after any deepcopy of it."""
    compiled = torch.compile(module._call_impl, **kwargs)
    eager = module._call_impl

    def call(*args, **kw):
        return eager(*args, **kw) if _EAGER_ONLY[0] else compiled(*args, **kw)
    module._compiled_call_impl = call
    return module


def compile_available(device):
    """-> (usable, reason): torch.compile needs CUDA here and the triton package (triton-windows on Windows)."""
    import importlib.util
    if device.type != "cuda":
        return False, "not on CUDA"
    if importlib.util.find_spec("triton") is None:
        return False, 'triton is not installed (Windows: pip install -U "triton-windows<3.9" for torch 2.14-2.15)'
    return True, ""


def setup_compile():
    """Dynamo settings for the compiled parts: room for every shape / module, eager fallback on errors."""
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    torch._dynamo.config.accumulated_cache_size_limit = max(torch._dynamo.config.accumulated_cache_size_limit, 1024)
    torch._dynamo.config.suppress_errors = True


# --- Learned loss weights -------------------------------------------------------------------------------------------
class LossHarmonizer(nn.Module):
    """HarmonyDream: each loss group gets weight 1 / sigma_i, sigma_i fitted on sum L_i / sigma_i + log(1 + sigma_i)."""
    GROUPS = ("pixel", "grounding", "kl", "rollout", "reward", "cont", "grid")

    def __init__(self, groups=GROUPS):
        super().__init__()
        self.groups = tuple(groups)
        self.log_sigma = nn.Parameter(torch.zeros(len(self.groups)))

    def weights(self):
        """{group: 1 / sigma} (detached)."""
        w = torch.exp(-self.log_sigma.detach())
        return {g: w[i] for i, g in enumerate(self.groups)}

    def scale_loss(self, values):
        """Loss of the scales given detached group losses; inactive groups (value <= 0) are skipped."""
        loss = self.log_sigma.new_zeros(())
        for i, g in enumerate(self.groups):
            v = values.get(g)
            if v is None:
                continue
            v = v.detach().float()
            s = self.log_sigma[i]
            loss = loss + (v > 0).float() * (v * torch.exp(-s) + F.softplus(s))
        return loss


# --- Phase timing ---------------------------------------------------------------------------------------------------
class PhaseTimer:
    """Wall time per update phase; syncs the GPU at each mark, so it only runs on metric updates."""

    def __init__(self, device):
        self.cuda = device.type == "cuda"
        self.on = False
        self.times = {}

    def _now(self):
        """Wall time after the GPU finished its queued work."""
        if self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def start(self, on=True, reset=True):
        """Starts timing (a no-op when on is False)."""
        self.on = on
        if on:
            if reset:
                self.times = {}
            self._last = self._now()

    def mark(self, name):
        """Adds the time since the last mark to phase `name`."""
        if self.on:
            now = self._now()
            self.times[name] = self.times.get(name, 0.0) + now - self._last
            self._last = now
