"""Model: every network of GQ-Dreamer in one file (latents, networks on s = [h, z], tokenizer, dynamics).

Sections, in dependency order:
  Latents     one-hot categorical latents: unimix log-probabilities, sampling, KL, entropy, class usage
  Networks    binned (two-hot / HL-Gauss) reward / value heads, continue and grid heads, actor, critic
  Tokenizer   GroundQuery fusion-transformer posterior, pixel decoder (transformer or conv), grounding heads
  Dynamics    transformer state-space model (TSSM), KV ring cache, prior head, CUDA-graphed dream step
How these networks run and learn is in gq_dreamer.py.
"""
import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from .game import BLOCK_COLS, BLOCK_ROWS, GRID_DIM
from .observations import (CNN_COLS, CNN_ROWS, IMG_H, IMG_W, MODE_DIM, NUM_MAP_IDS, N_CNN_TOKENS, N_TIERS, STATUS_DIM,
                           TILE_PX, autocast_dtype, frame_input, frame_rgb, n_patches, patch_dim, patch_grid,
                           patch_pixel_mask, patchify, ram_fields, sprite_pixel_mask, sprite_tiles, status_input,
                           unpatchify)
from .train_utils import eager_modules, eager_only


# === Latents ========================================================================================================
# One-hot categorical latents (Dreamer): unimix log-probabilities, sampling, KL, entropy and class usage.


def unimix_logprobs(logits, unimix=0.01):
    """Logits [..., K] -> fp32 log-probabilities of (1 - unimix) softmax + unimix / K."""
    logits = logits.float()
    if not unimix:
        return torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    probs = (1.0 - unimix) * probs + unimix / logits.shape[-1]
    return torch.log(probs)


def gumbel_like(logp):
    """Gumbel(0, 1) noise shaped like logp."""
    u = torch.rand_like(logp, dtype=torch.float32).clamp_(1e-6, 1.0 - 1e-6)
    return -torch.log(-torch.log(u))


def sample_onehot(logp, straight_through=True, gumbel=None):
    """Log-probabilities [..., L, K] -> one-hot sample (straight-through gradient by default)."""
    g = gumbel_like(logp) if gumbel is None else gumbel
    onehot = F.one_hot((logp.detach().float() + g).argmax(-1), logp.shape[-1]).float()
    if not straight_through:
        return onehot
    probs = logp.float().exp()
    return onehot + probs - probs.detach()


def kl_categorical(logp, logq):
    """KL(p || q) of [..., L, K] log-probabilities, summed over classes and categoricals -> [...] nats."""
    return (logp.exp() * (logp - logq)).sum((-1, -2))


def entropy_categorical(logp):
    """Entropy of [..., L, K] log-probabilities, summed over categoricals -> [...] nats."""
    return -(logp.exp() * logp).sum((-1, -2))


@torch.no_grad()
def class_usage(z, n_latents, classes, mask=None):
    """Batch class perplexity / K, averaged over categoricals (1 = all classes used, 1/K = collapsed)."""
    z = z.float().reshape(-1, n_latents, classes)
    w = torch.ones(z.shape[0], device=z.device) if mask is None else mask.float().reshape(-1)
    p = (z * w[:, None, None]).sum(0) / w.sum().clamp_min(1.0)
    h = -(p * p.clamp_min(1e-12).log()).sum(-1)
    return (h.exp() / classes).mean()


# === Networks =======================================================================================================
# MLP heads on the agent state s = [h, z]: binned (two-hot / HL-Gauss) reward / value heads, continue and grid heads,
# actor and critic.


def mlp(in_dim, out_dim, hidden=1024, layers=3, act=nn.SiLU):
    """MLP of `layers` hidden Linear-LayerNorm-act layers."""
    mods, d = [], in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), act()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


def symlog(x):
    """sign(x) * log(|x| + 1)."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    """Inverse of symlog."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


VALUE_TARGETS = ("twohot", "hl_gauss")


class TwoHotEncoding:
    """Symlog binned regression over `num_bins` bins: decodes the softmax-weighted mean of the bin values, trained by
    cross-entropy to a "twohot" target (DreamerV3: the two bins around symlog(y), linearly weighted) or an "hl_gauss"
    target (Farebrother et al. 2024: the mass of N(symlog(y), sigma^2) in each bin, sigma = sigma_ratio x bin
    width)."""

    def __init__(self, min=-10, max=10, num_bins=255, device="cuda", target="twohot", sigma_ratio=0.75):
        if target not in VALUE_TARGETS:
            raise ValueError(f"value target {target!r}: one of {VALUE_TARGETS}")
        self.num_bins, self.target = num_bins, target
        self.bins = torch.linspace(min, max, num_bins, device=device)
        self.bin_values = symexp(self.bins)
        self.sigma = float(sigma_ratio) * (max - min) / (num_bins - 1)
        self.edges = 0.5 * (self.bins[1:] + self.bins[:-1])           # inner bin edges [num_bins - 1]

    def encode(self, x):
        """Scalars [...] -> target distributions [..., num_bins] that each sum to 1."""
        if self.target == "hl_gauss":
            return self._hl_gauss(x)
        x = torch.clamp(symlog(x.float()), self.bins[0], self.bins[-1])
        pos = (x - self.bins[0]) / (self.bins[1] - self.bins[0])
        low = torch.clamp(torch.floor(pos).long(), 0, self.num_bins - 2)
        w_high = pos - low
        two_hot = torch.zeros(*x.shape, self.num_bins, device=x.device)
        two_hot.scatter_(-1, low.unsqueeze(-1), (1.0 - w_high).unsqueeze(-1))
        two_hot.scatter_(-1, (low + 1).unsqueeze(-1), w_high.unsqueeze(-1))
        return two_hot

    def _hl_gauss(self, x):
        """HL-Gauss target p_i = Phi((e_{i+1} - y) / sigma) - Phi((e_i - y) / sigma), y = symlog(x), outer edges open
        (e_0 = -inf, e_num_bins = +inf), normalised."""
        y = symlog(x.float())
        cdf = torch.special.ndtr((self.edges - y[..., None]) / self.sigma)            # [..., num_bins - 1]
        zero, one = torch.zeros_like(cdf[..., :1]), torch.ones_like(cdf[..., :1])
        p = (torch.cat([cdf, one], dim=-1) - torch.cat([zero, cdf], dim=-1)).clamp_min(0.0)
        return p / p.sum(-1, keepdim=True).clamp_min(1e-12)

    def decode(self, logits):
        """Logits [..., num_bins] -> expected value [..., 1]."""
        return torch.sum(torch.softmax(logits.float(), dim=-1) * self.bin_values, dim=-1, keepdim=True)

    def loss(self, logits, target, weights=None):
        """Weighted cross-entropy against the target distribution of the scalars `target`."""
        with torch.no_grad():
            th = self.encode(target)
        ce = -torch.sum(th * torch.log_softmax(logits.float(), dim=-1), dim=-1)
        if weights is None:
            return ce.mean()
        return (ce * weights).sum() / weights.sum().clamp(min=1e-6)


class TwoHotHead(nn.Module):
    """Two-hot regression head (rewards, curiosity), zero-initialised output."""

    def __init__(self, in_dim, num_bins=255, hidden=1024, zero_bias_bin=None):
        super().__init__()
        self.net = mlp(in_dim, num_bins, hidden=hidden, layers=2)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        if zero_bias_bin is not None:                                  # start confidently at the zero bin
            with torch.no_grad():
                self.net[-1].bias[zero_bias_bin] = 5.0

    def forward(self, x):
        return self.net(x)


class ContinueHead(nn.Module):
    """Bernoulli logit of 'Mario is still alive in this state'."""

    def __init__(self, in_dim, hidden=1024):
        super().__init__()
        self.net = mlp(in_dim, 1, hidden=hidden, layers=2)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 4.0)                      # p(alive) ~ 0.98 at init

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GridHead(nn.Module):
    """Logits of the 5x5 explored grid (its centre gates curiosity in dreams)."""

    def __init__(self, in_dim, hidden=1024):
        super().__init__()
        self.net = mlp(in_dim, GRID_DIM, hidden=hidden, layers=2)

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    """Categorical policy over the actions with a 1% uniform mix."""

    def __init__(self, action_dim, in_dim, mlp_dim=1024):
        super().__init__()
        self.action_dim = action_dim
        self.net = mlp(in_dim, action_dim, hidden=mlp_dim, layers=3)
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def probs(self, x, unimix=True):
        """Action probabilities [..., A]; with unimix (training) 1% of the mass is spread uniformly."""
        probs = torch.softmax(self.net(x).float(), dim=-1)
        return 0.99 * probs + 0.01 / self.action_dim if unimix else probs

    def forward(self, x, greedy=False, return_probs=False, unimix=True):
        """-> (one-hot action, log-prob, entropy[, probs])."""
        probs = self.probs(x, unimix)
        dist = OneHotCategorical(probs=probs, validate_args=False)     # validation would sync the GPU
        a = F.one_hot(probs.argmax(-1), self.action_dim).float() if greedy else dist.sample()
        if return_probs:
            return a, dist.log_prob(a), dist.entropy(), probs
        return a, dist.log_prob(a), dist.entropy()


class Critic(nn.Module):
    """Two-hot value head."""

    def __init__(self, in_dim, bins=255, mlp_dim=1024):
        super().__init__()
        self.net = mlp(in_dim, bins, hidden=mlp_dim, layers=3)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


# === Tokenizer ======================================================================================================
# GroundQuery tokenizer: fusion-transformer posterior read out by latent queries, pixel decoder (transformer or
# conv), grounding heads (they read z only, so level and status have to be stored in the latent itself).


# --- Building blocks ------------------------------------------------------------------------------------------------
class ConvResBlock(nn.Module):
    """GroupNorm-SiLU-conv x2 residual block, identity at init."""

    def __init__(self, ch, groups=8):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(groups, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
                                 nn.GroupNorm(groups, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


def _sac_policy(ctx, op, *args, **kwargs):
    """Selective activation checkpointing: keep conv / matmul outputs, recompute everything else (norms,
    activations, upsampling, adds, the loss) in the backward pass."""
    if op in (torch.ops.aten.convolution.default, torch.ops.aten.mm.default, torch.ops.aten.addmm.default):
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


SAC_CONTEXT = partial(create_selective_checkpoint_contexts, _sac_policy)


class _DropSacCompileNotice(logging.Filter):
    """Drops torch.compile's notice that context_fn (the decoder's selective checkpointing) is used under compile, a
    reminder to avoid in-place ops in the checkpointed region; every other torch warning still shows."""

    def filter(self, record):
        return "context_fn is passed to torch.utils.checkpoint" not in record.getMessage()


_SAC_NOTICE_FILTER = _DropSacCompileNotice()
logging.getLogger("torch._higher_order_ops.wrap").addFilter(_SAC_NOTICE_FILTER)   # where torch logs it
for _h in logging.getLogger("torch").handlers:                                       # and any copy that propagates
    _h.addFilter(_SAC_NOTICE_FILTER)

# sub-pixel form of nearest x3 + 3x3 conv: M[a, r, d] = 1 when tap d of output phase a reads low-res offset r - 1
_SUBPIX_M = torch.zeros(3, 3, 3)
for _a in range(3):
    for _d in range(3):
        _SUBPIX_M[_a, (_a + _d - 1) // 3 + 1, _d] = 1.0


class SubPixelConv2d(nn.Conv2d):
    """A 3x3 conv (padding 1) applied to a nearest x3 upsampling of its input, computed without the upsampling: the
    same parameters give a 3x3 conv at the low resolution to 9 x out_channels (each output phase sums the taps that
    read the same low-res pixel) followed by PixelShuffle(3). Exact (float order aside); the forward takes the LOW-RES
    input. Same 212M MACs per frame at 32x40 -> 96x120, but a GEMM with 9x the output channels and no full-resolution
    upsampled tensor in the forward, recomputation or backward."""

    def forward(self, x):
        co, ci = self.weight.shape[:2]
        with torch.autocast(x.device.type, enabled=False):            # tap sums in fp32: one bf16 rounding later
            m = _SUBPIX_M.to(self.weight.device)
            w = torch.einsum("ard,bse,oide->oabirs", m, m, self.weight.float()).reshape(co * 9, ci, 3, 3)
        b = self.bias.repeat_interleave(9) if self.bias is not None else None
        return F.pixel_shuffle(F.conv2d(x, w, b, padding=1), 3)


class PosInjectLayer(nn.TransformerEncoderLayer):
    """Pre-norm encoder layer (built on nn.TransformerEncoderLayer) whose attention input gets a fixed position term:
    x + SA(LN1(x) + pos), then + FF(LN2(x)). As in DETR, pos is re-added in every layer instead of once at the input.
    It is added after the LayerNorm, so it keeps its size next to the normalised content however large the residual
    stream grows. It enters the queries, keys and values: a latent query that finds a sprite by its content also reads
    where the sprite is. pos = None: the plain layer.

    swiglu: the TSSM's feed-forward, down(silu(gate(x)) * up(x)) without biases, hidden dim_feedforward (instead of
    the GELU MLP linear2(gelu(linear1(x))))."""

    def __init__(self, d_model, nhead, dim_feedforward, *args, swiglu=False, **kwargs):
        super().__init__(d_model, nhead, dim_feedforward, *args, **kwargs)
        self.swiglu = swiglu
        self.fast_attn = False                    # set by FusionEncoder: SDPA on views instead of nn.MultiheadAttention
        if swiglu:
            del self.linear1, self.linear2
            self.gate_proj = nn.Linear(d_model, dim_feedforward, bias=False)
            self.up_proj = nn.Linear(d_model, dim_feedforward, bias=False)
            self.down_proj = nn.Linear(dim_feedforward, d_model, bias=False)

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        """Self-attention. fast_attn: nn.MultiheadAttention's math with the same parameters, but scaled_dot_product_
        attention on views instead of its transposes and contiguous copies."""
        if not self.fast_attn or attn_mask is not None or key_padding_mask is not None:
            return super()._sa_block(x, attn_mask, key_padding_mask, is_causal)
        a = self.self_attn
        n, t, e = x.shape
        qkv = F.linear(x, a.in_proj_weight, a.in_proj_bias).view(n, t, 3, a.num_heads, e // a.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(n, t, e)
        return self.dropout1(F.linear(o, a.out_proj.weight, a.out_proj.bias))

    def _ff_block(self, x):
        if not self.swiglu:
            return super()._ff_block(x)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def forward(self, x, pos=None):
        h = self.norm1(x)
        if pos is not None:
            h = h + pos.to(h.dtype)
        x = x + self._sa_block(h, None, None)
        return x + self._ff_block(self.norm2(x))

    def forward_first(self, x, pos, n):
        """forward(x, pos)[:, :n], computing queries, attention output and feed-forward for the first n tokens only
        (keys / values from every token); the encoder's last layer, whose other outputs are never read."""
        h = self.norm1(x)
        if pos is not None:
            h = h + pos.to(h.dtype)
        a = self.self_attn
        b, t, e = x.shape
        nh = a.num_heads
        w, bias = a.in_proj_weight, a.in_proj_bias
        q = F.linear(h[:, :n], w[:e], None if bias is None else bias[:e]).view(b, n, nh, e // nh).transpose(1, 2)
        kv = F.linear(h, w[e:], None if bias is None else bias[e:]).view(b, t, 2, nh, e // nh)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, e)
        x = x[:, :n] + self.dropout1(F.linear(o, a.out_proj.weight, a.out_proj.bias))
        return x + self._ff_block(self.norm2(x))


def _transformer(dim, depth, heads, layer_cls=nn.TransformerEncoderLayer, ffn=None, **layer_kwargs):
    """Pre-norm transformer encoder, GELU MLP 4 x dim by default (layer_cls: PosInjectLayer for position injection in
    every layer, and swiglu=True for its SwiGLU feed-forward with hidden ffn)."""
    layer = layer_cls(dim, heads, ffn or dim * 4, dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
                      **layer_kwargs)
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)


def sincos_2d(rows, cols, dim, min_wavelength=4.0, max_wavelength=64.0):
    """Fixed 2D sin-cos embedding [n, dim] of token centres (rows[i], cols[i]) in game tiles (the screen is 8 x 10
    tiles). Half of the channels encode the row and half the column, each as dim / 4 (sin, cos) pairs whose
    wavelengths are spaced geometrically from min_wavelength to max_wavelength tiles. Every value is in [-1, 1]
    and the vector's norm is sqrt(dim / 2)."""
    assert dim % 4 == 0, "sincos_2d needs dim divisible by 4"
    n = dim // 4
    wl = min_wavelength * (max_wavelength / min_wavelength) ** (torch.arange(n, dtype=torch.float64) / max(n - 1, 1))
    omega = 2.0 * math.pi / wl

    def axis(p):
        a = torch.as_tensor(p, dtype=torch.float64)[:, None] * omega[None]
        return torch.cat([a.sin(), a.cos()], dim=1)
    return torch.cat([axis(rows), axis(cols)], dim=1).float()


def image_token_centres(stem, patch_tiles):
    """(rows, cols) of the image tokens' centres in game tiles, in token order (row-major): "linear" = k x k-tile
    patches (k = patch_tiles), "cnn" = the 8 x 5 grid of 1 x 2-tile tokens."""
    if stem == "cnn":
        r, c, h, w = CNN_ROWS, CNN_COLS, 1, 2
    else:
        r, c, _ = patch_grid(patch_tiles)
        h = w = patch_tiles
    rr, cc = torch.meshgrid(torch.arange(r, dtype=torch.float64), torch.arange(c, dtype=torch.float64), indexing="ij")
    return (rr * h + h / 2).flatten(), (cc * w + w / 2).flatten()


def _small_mlp(in_dim, out_dim, hidden=256):
    """One-hidden-layer SiLU MLP."""
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))


def bce_floor(target):
    """Entropy of the target colours: the lowest BCE a perfect reconstruction reaches (fp32)."""
    t = target.float().clamp(1e-6, 1.0 - 1e-6)
    return -(t * t.log() + (1.0 - t) * (1.0 - t).log())


# --- Encoder (posterior) --------------------------------------------------------------------------------------------
class CNNStem(nn.Module):
    """Small CNN: frame [N, 3, 96, 120] -> 8x5 grid of tokens [N, 40, dim], one token per 1x2 game tiles (12x24 px)
    with overlapping receptive fields. 96x120 -> 48x60 -> 24x30 (two stride-2 stages), then a (3, 6) / (3, 6)
    convolution to 8x5."""

    def __init__(self, dim=192, chans=(32, 64), groups=8):
        super().__init__()
        c1, c2 = chans
        self.net = nn.Sequential(
            nn.Conv2d(3, c1, 4, stride=2, padding=1), nn.GroupNorm(groups, c1), nn.SiLU(),        # 48x60
            nn.Conv2d(c1, c1, 3, padding=1), nn.GroupNorm(groups, c1), nn.SiLU(),
            nn.Conv2d(c1, c2, 4, stride=2, padding=1), nn.GroupNorm(groups, c2), nn.SiLU(),       # 24x30
            nn.Conv2d(c2, c2, 3, padding=1), nn.GroupNorm(groups, c2), nn.SiLU(),
            nn.Conv2d(c2, dim, kernel_size=(IMG_H // 4 // CNN_ROWS, IMG_W // 4 // CNN_COLS),
                      stride=(IMG_H // 4 // CNN_ROWS, IMG_W // 4 // CNN_COLS)))                  # 8x5

    def forward(self, img):
        return self.net(img).flatten(2).transpose(1, 2)                              # [N, 40, dim], row-major


class FusionEncoder(nn.Module):
    """One transformer over the image tokens + grid / level / status tokens, read out by n_latents learned queries.
    Image tokens: stem "linear" = a linear embedding of the k x k-tile patches (k = patch_tiles: 80 patches for k = 1),
    taken from the native 128x160 screen (native=True) or the 96x120 frame; stem "cnn" = a small CNN over the 96x120
    frame -> 40 tokens (8x5, 1x2 tiles each). pos_inject: a fixed 2D sin-cos position of every image token is added
    to the attention input of every layer (PosInjectLayer); the latent queries and side tokens get none.
    last_queries: the last layer computes only the latent queries' outputs (the only ones read; same result)."""
    TYPES = ("image", "grid", "level", "status")
    STEM_CHUNK = 1024                                                              # frames per checkpointed CNN pass
    LAYER_CHUNK = 2048                                                             # frames per checkpointed layer pass

    def __init__(self, dim=256, depth=4, heads=8, n_latents=32, classes=32, grad_checkpoint=True, stem="linear",
                 patch_tiles=2, native=True, pos_inject=True, swiglu=True, ffn=682, fast_attn=False,
                 last_queries=False):
        super().__init__()
        if stem not in ("cnn", "linear"):
            raise ValueError(f"unknown encoder stem {stem!r}: use 'cnn' or 'linear'")
        if stem == "cnn" and native:
            raise ValueError("the CNN stem reads the 96x120 frame: use enc_input='model' with enc_stem='cnn'")
        self.n_latents, self.classes, self.stem_type = n_latents, classes, stem
        self.patch_tiles, self.native = patch_tiles, native
        self.grad_checkpoint = grad_checkpoint
        self.last_queries = last_queries                                          # last layer: query outputs only
        if stem == "cnn":
            self.stem = CNNStem(dim)
            self.n_image_tokens = N_CNN_TOKENS
        else:
            self.patch_embed = nn.Linear(patch_dim(patch_tiles, native), dim)      # linear patch embedding
            self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.n_image_tokens = n_patches(patch_tiles)
        self.image_pos = nn.Parameter(torch.randn(1, self.n_image_tokens, dim) * 0.02)
        self.latent_tokens = nn.Parameter(torch.randn(1, n_latents, dim) * 0.02)  # the latent queries
        self.grid_in = _small_mlp(GRID_DIM, dim, dim)
        self.level_emb = nn.Embedding(NUM_MAP_IDS, dim)
        nn.init.normal_(self.level_emb.weight, std=0.02)
        self.mode_in = nn.Linear(MODE_DIM, dim)
        self.status_in = _small_mlp(STATUS_DIM, dim, dim)
        self.type_emb = nn.Parameter(torch.randn(len(self.TYPES), dim) * 0.02)
        self.missing = nn.Parameter(torch.randn(3, dim) * 0.02)                   # dropped grid / level / status
        self.transformer = (_transformer(dim, depth, heads, PosInjectLayer, ffn, swiglu=True) if swiglu
                            else _transformer(dim, depth, heads, PosInjectLayer))
        for layer in self.transformer.layers:
            layer.fast_attn = fast_attn
        self._ckpt_runs = None                                                    # compiled checkpointed layers
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, classes)                                  # shared by all queries
        # position injected in every layer: sin-cos of the image tokens' screen position, 0 on the other tokens.
        # Not saved in checkpoints (a fixed function of the token layout).
        inject = None
        if pos_inject:
            inject = torch.zeros(1, n_latents + self.n_image_tokens + len(self.TYPES) - 1, dim)
            inject[0, n_latents:n_latents + self.n_image_tokens] = sincos_2d(*image_token_centres(stem, patch_tiles),
                                                                             dim)
        self.register_buffer("inject_pos", inject, persistent=False)

    def _layer_fns(self):
        """Each layer's callable (x, pos) -> x; with last_queries the last one returns the latent queries only."""
        fns = list(self.transformer.layers)
        if self.last_queries:
            fns[-1] = partial(fns[-1].forward_first, n=self.n_latents)
        return fns

    def compile_layers(self):
        """torch.compile of each checkpointed layer pass (training only; the recomputation is part of the compiled
        backward, the no-grad passes stay eager)."""
        def runner(fn):
            def run(x, pos):
                return checkpoint(fn, x, pos, use_reentrant=False)
            return run
        self._ckpt_runs = [torch.compile(runner(fn)) for fn in self._layer_fns()]

    def _image_tokens(self, img, hide):
        """Frames in [-1, 1] -> image tokens [N, n_image_tokens, dim]; hide [N, n_patches] hides background patches."""
        if self.stem_type == "linear":
            tok = self.patch_embed(patchify(img, self.patch_tiles))
            if hide is not None:
                tok = torch.where(hide[..., None], self.mask_token.to(tok.dtype), tok)
            return tok
        if hide is not None:                                                       # hidden patches -> neutral grey
            img = img.masked_fill(patch_pixel_mask(hide, self.patch_tiles), 0.0)   # before the CNN: nothing leaks
        if not (self.grad_checkpoint and torch.is_grad_enabled()):
            return self.stem(img)
        # activations recomputed in the backward pass, chunk by chunk (peak memory of one chunk, not of the batch)
        return torch.cat([checkpoint(self.stem, img[i:i + self.STEM_CHUNK], use_reentrant=False)
                          for i in range(0, img.shape[0], self.STEM_CHUNK)])

    def forward(self, img, grid, map_id, mode, status, hide=None, drop=None):
        """img [N, 3, H, W] in [-1, 1] (observations.frame_input) -> posterior logits [N, L, K]; hide [N, n_patches]
        hides background patches (masking branch), drop [N, 3] side tokens."""
        N = img.shape[0]
        tok = self._image_tokens(img, hide)
        dt = tok.dtype
        tok = tok + self.image_pos.to(dt) + self.type_emb[0].to(dt)
        side = torch.stack([self.grid_in(grid.to(dt)).to(dt),
                            (self.level_emb(map_id.long()) + self.mode_in(mode.to(dt))).to(dt),
                            self.status_in(status.to(dt)).to(dt)], dim=1)
        if drop is not None:
            side = torch.where(drop[..., None], self.missing.to(dt).expand(N, -1, -1), side)
        side = side + self.type_emb[1:].to(dt)
        x = torch.cat([self.latent_tokens.expand(N, -1, -1).to(dt), tok, side], dim=1)   # [N, L + image + 3, dim]
        ckpt = self.grad_checkpoint and torch.is_grad_enabled() and x.requires_grad
        pos = self.inject_pos                                                    # [1, tokens, dim] or None
        for li, fn in enumerate(self._layer_fns()):
            if not ckpt:
                x = fn(x, pos)
            else:   # each layer recomputed in the backward pass, in chunks of frames (bounded peak memory)
                run = (self._ckpt_runs[li] if self._ckpt_runs is not None and not eager_only()
                       else partial(checkpoint, fn, use_reentrant=False))
                x = torch.cat([run(x[i:i + self.LAYER_CHUNK], pos) for i in range(0, N, self.LAYER_CHUNK)])
        return self.to_logits(self.norm(x[:, :self.n_latents])).float()


# --- Decoders -------------------------------------------------------------------------------------------------------
class PixelDecoder(nn.Module):
    """Flat one-hot z -> RGB logits [N, 3, 96, 120]: Linear to 4x5, 3 ConvTranspose x2 stages (32x40), then
    nearest x3 + conv (subpixel: the same conv computed at 32x40 + PixelShuffle, SubPixelConv2d)."""

    def __init__(self, latent_dim=1024, chans=(128, 128, 96, 64, 32), res_stages=(1, 2, 3), subpixel=False):
        super().__init__()
        assert len(chans) == 5, "the decoder has 4 upsampling stages: chans = (c0, c1, c2, c3, c4)"
        self.c0 = chans[0]
        self.fc = nn.Linear(latent_dim, chans[0] * 4 * 5)
        layers = []
        for i in range(4):
            cin, cout = chans[i], chans[i + 1]
            layers += [nn.GroupNorm(8, cin), nn.SiLU()]
            if i < 3:
                layers.append(nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1))
            elif subpixel:                                             # 32x40 -> 96x120, same parameters / keys
                layers += [nn.Identity(), SubPixelConv2d(cin, cout, 3, padding=1)]
            else:                                                      # 32x40 -> 96x120
                layers += [nn.Upsample(scale_factor=3, mode="nearest"), nn.Conv2d(cin, cout, 3, padding=1)]
            if i + 1 in res_stages:
                layers.append(ConvResBlock(cout))
        layers += [nn.GroupNorm(8, chans[-1]), nn.SiLU(), nn.Conv2d(chans[-1], 3, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(self.fc(z.flatten(1)).view(-1, self.c0, 4, 5))

    @staticmethod
    def to_image(logits):
        """Logits [N, 3, H, W] -> image [N, H, W, 3] in [0, 1]."""
        return torch.sigmoid(logits.float()).permute(0, 2, 3, 1)


class TransformerPixelDecoder(nn.Module):
    """Flat one-hot z -> RGB logits [N, 3, 96, 120] with a transformer (MAE / TiTok style).

    Each categorical becomes a token (its own class embeddings, so class k means something different in every
    categorical). One learned query per output patch (k x k tiles: 20 queries of 24x24 px for k = 2) attends to them and
    to the other queries; a linear layer turns each query into its patch's pixels, and a zero-initialised residual
    conv smooths the patch seams."""

    def __init__(self, n_latents=32, classes=32, dim=256, depth=4, heads=8, patch_tiles=2):
        super().__init__()
        self.n_latents, self.classes, self.patch_tiles = n_latents, classes, patch_tiles
        self.latent_emb = nn.Parameter(torch.randn(n_latents, classes, dim) * 0.02)   # per categorical and class
        self.latent_pos = nn.Parameter(torch.randn(1, n_latents, dim) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, n_patches(patch_tiles), dim) * 0.02)  # one per output patch
        self.transformer = _transformer(dim, depth, heads)
        self.norm = nn.LayerNorm(dim)
        self.to_pixels = nn.Linear(dim, patch_dim(patch_tiles))
        self.refine = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 3, 3, padding=1))
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, z):
        N = z.shape[0]
        zz = z.reshape(N, self.n_latents, self.classes)
        lat = torch.einsum("nlk,lkd->nld", zz.to(self.latent_emb.dtype), self.latent_emb) + self.latent_pos
        x = torch.cat([lat, self.queries.expand(N, -1, -1)], dim=1)
        x = self.norm(self.transformer(x)[:, self.n_latents:])
        logits = unpatchify(self.to_pixels(x), self.patch_tiles)                  # [N, 3, 96, 120]
        return logits + self.refine(logits)

    @staticmethod
    def to_image(logits):
        """Logits [N, 3, H, W] -> image [N, H, W, 3] in [0, 1]."""
        return torch.sigmoid(logits.float()).permute(0, 2, 3, 1)


class GroundingHeads(nn.Module):
    """Small MLPs decoding level and status from the flat latent z, so the latent stores them explicitly. (The
    explored grid depends on the episode's history: its head reads s = [h, z] and lives in the agent.)"""

    def __init__(self, latent_dim, hidden=256):
        super().__init__()
        self.level = _small_mlp(latent_dim, NUM_MAP_IDS + MODE_DIM, hidden)
        self.status = _small_mlp(latent_dim, N_TIERS + 3, hidden)   # tier logits, star, facing, lives

    def losses(self, zf, map_id, ram, mask=None):
        """Grounding losses {"level", "status"} and accuracies of flat latents zf [..., D]."""
        zf = zf.float()
        lead = zf.shape[:-1]
        m = torch.ones(lead, device=zf.device) if mask is None else mask.float()
        denom = m.sum().clamp(min=1.0)
        rf = ram_fields(ram)

        lv = self.level(zf).float()
        lvl_logits, mode_logits = lv[..., :NUM_MAP_IDS], lv[..., NUM_MAP_IDS:]
        level_l = (F.cross_entropy(lvl_logits.reshape(-1, NUM_MAP_IDS), map_id.long().reshape(-1), reduction="none")
                   .view(lead)
                   + F.binary_cross_entropy_with_logits(mode_logits, rf["mode"], reduction="none").mean(-1))

        st = self.status(zf).float()
        tier_logits = st[..., :N_TIERS]
        star_l, face_l, lives_p = st[..., N_TIERS:N_TIERS + 3].unbind(-1)
        status_l = (F.cross_entropy(tier_logits.reshape(-1, N_TIERS), rf["tier"].reshape(-1), reduction="none")
                    .view(lead)
                    + F.binary_cross_entropy_with_logits(star_l, rf["star"], reduction="none")
                    + F.binary_cross_entropy_with_logits(face_l, rf["facing"], reduction="none")
                    + (lives_p - rf["lives"]).pow(2))

        losses = {k: (v * m).sum() / denom for k, v in (("level", level_l), ("status", status_l))}
        with torch.no_grad():
            acc = {"level_acc": ((lvl_logits.argmax(-1) == map_id.long()).float() * m).sum() / denom,
                   "tier_acc": ((tier_logits.argmax(-1) == rf["tier"]).float() * m).sum() / denom}
        return losses, acc

    @torch.no_grad()
    def predict(self, zf):
        """Flat latent -> (map id, power-up tier)."""
        return self.level(zf.float())[..., :NUM_MAP_IDS].argmax(-1), self.status(zf.float())[..., :N_TIERS].argmax(-1)


# --- Tokenizer ------------------------------------------------------------------------------------------------------
class Tokenizer(nn.Module):
    """Encoder (posterior) + pixel decoder + grounding heads (level / status) around the n_latents x classes
    one-hot latent; the decoder and the grounding heads read z only."""

    def __init__(self, n_latents=32, classes=32, enc_dim=256, enc_depth=4, enc_heads=8,
                 dec_chans=(128, 128, 96, 64, 32), dec_res=(1, 2, 3), head_hidden=256, unimix=0.01,
                 grad_checkpoint=True, enc_stem="linear", enc_patch_tiles=2, enc_native=True,
                 decoder="transformer", dec_dim=256, dec_depth=4, dec_heads=8, dec_patch_tiles=2, enc_pos_inject=True,
                 enc_swiglu=True, enc_ffn=682, enc_fast_attn=False, dec_subpixel=False, dec_sac=False,
                 dec_targets_once=False, enc_last_queries=False):
        super().__init__()
        self.n_latents, self.classes, self.unimix = n_latents, classes, unimix
        self.latent_dim = n_latents * classes
        self.dec_sac, self.dec_targets_once = dec_sac, dec_targets_once
        self._chunk_run = None                                         # compiled checkpointed decoder chunk
        self.encoder = FusionEncoder(enc_dim, enc_depth, enc_heads, n_latents, classes, grad_checkpoint, enc_stem,
                                     enc_patch_tiles, enc_native, enc_pos_inject, enc_swiglu, enc_ffn, enc_fast_attn,
                                     enc_last_queries)
        if decoder == "transformer":
            self.decoder = TransformerPixelDecoder(n_latents, classes, dec_dim, dec_depth, dec_heads,
                                                   dec_patch_tiles)
        elif decoder == "conv":
            self.decoder = PixelDecoder(self.latent_dim, dec_chans, dec_res, subpixel=dec_subpixel)
        else:
            raise ValueError(f"unknown decoder {decoder!r}: use 'transformer' or 'conv'")
        self.heads = GroundingHeads(self.latent_dim, head_hidden)

    def _inputs(self, obs):
        """Observation dict -> encoder inputs (frames in [-1, 1], grid, map id, mode, status)."""
        screen = obs["screen"]
        img = frame_input(screen, obs["palette"], autocast_dtype(screen.device.type), native=self.encoder.native)
        return img, obs["grid"], obs["map_id"], ram_fields(obs["ram"])["mode"], status_input(obs)

    def posterior(self, obs, hide=None, drop=None):
        """Observations -> log-probabilities [N, L, K] of q(z | o) (fp32, unimix)."""
        return unimix_logprobs(self.encoder(*self._inputs(obs), hide, drop), self.unimix)

    def encode(self, obs, straight_through=False):
        """Observations -> (flat one-hot sample [N, L*K], log-probabilities [N, L, K])."""
        logp = self.posterior(obs)
        return sample_onehot(logp, straight_through).flatten(1), logp

    def decode(self, z):
        """Flat latent -> RGB logits [N, 3, 96, 120]."""
        return self.decoder(z)

    @staticmethod
    def pixel_loss(logits, rgb, blocks, sprite_weight=5.0):
        """Sprite-weighted BCE summed over the frame, minus its entropy floor (0 = perfect) -> (loss, stats)."""
        bce = F.binary_cross_entropy_with_logits(logits.float(), rgb, reduction="none")
        excess = (bce - bce_floor(rgb)).sum(1)                                     # [N, H, W]
        sprite = sprite_pixel_mask(blocks).float()
        w = 1.0 + (sprite_weight - 1.0) * sprite
        loss = (excess * w).sum() / w.sum() * (IMG_H * IMG_W)
        with torch.no_grad():
            ex = excess.detach() / 3.0                                             # per colour value
            stats = {"recon_excess": ex.mean(),
                     "recon_sprite_excess": (ex * sprite).sum() / sprite.sum().clamp_min(1.0)}
        return loss, stats

    DEC_CHUNK = 512                                                        # frames per checkpointed decoder pass

    @torch.no_grad()
    def recon_targets(self, obs, sprite_weight=5.0, chunk=1024):
        """Pixel-loss targets of the frames of obs, computed once per update outside the checkpoint: fp32 RGB
        [N, 3, 96, 120], and per game tile [N, 8, 10] the sprite mask, the loss weight and the BCE entropy floor
        (float64)."""
        n = obs["screen"].shape[0]
        rgb = torch.empty(n, 3, IMG_H, IMG_W, device=obs["screen"].device)
        floor = torch.empty(n, BLOCK_ROWS, BLOCK_COLS, device=rgb.device, dtype=torch.float64)
        for i in range(0, n, chunk):
            r = frame_rgb(obs["screen"][i:i + chunk], obs["palette"][i:i + chunk])
            rgb[i:i + chunk] = r
            floor[i:i + chunk] = _tile_sum(bce_floor(r).sum(1))
        sprite = sprite_tiles(obs["blocks"]).float()
        return {"rgb": rgb, "sprite": sprite, "w": 1.0 + (sprite_weight - 1.0) * sprite, "floor": floor}

    @staticmethod
    def subset_targets(targets, idx):
        """The recon_targets of a subset of the frames."""
        return {k: v[idx] for k, v in targets.items()}

    def _chunk_ckpt(self, *args):
        """One checkpointed decoder chunk (dec_sac: conv outputs kept, the rest recomputed in the backward)."""
        fn = self._chunk_recon if self.dec_targets_once else self._chunk_recon_legacy
        sac = self.dec_sac and not eager_only()                        # eager_modules(): re-traversable graph
        return checkpoint(fn, *args, use_reentrant=False, **({"context_fn": SAC_CONTEXT} if sac else {}))

    def compile_decoder(self):
        """torch.compile of the checkpointed chunk pass (the selective recomputation becomes part of the compiled
        backward)."""
        self._chunk_run = torch.compile(self._chunk_ckpt)

    def recon_loss(self, z, obs, sprite_weight=5.0, targets=None):
        """pixel_loss of decoding flat latents z [N, L*K] against the frames of obs (screen / palette / blocks), chunk
        by chunk: each chunk's decoder pass and loss are recomputed in the backward pass, so neither the decoder
        activations nor the logits of all N frames are stored (N can be the whole batch). dec_targets_once: the targets
        come from recon_targets (given, or computed here once); sum(excess * w) = sum(bce * w) - sum(floor * w) with
        the floor constant, so the loss and its gradient are unchanged."""
        if not self.dec_targets_once:
            return self._recon_loss_legacy(z, obs, sprite_weight)
        t = targets if targets is not None else self.recon_targets(obs, sprite_weight)
        ckpt = torch.is_grad_enabled() and z.requires_grad
        run = ((self._chunk_ckpt if eager_only() else self._chunk_run or self._chunk_ckpt) if ckpt
               else self._chunk_recon)
        num = tot = spr = 0.0
        for i in range(0, z.shape[0], self.DEC_CHUNK):
            j = i + self.DEC_CHUNK
            out = run(z[i:j], t["rgb"][i:j], t["sprite"][i:j], t["w"][i:j])
            num, tot, spr = num + out[0], tot + out[1], spr + out[2]
        px = TILE_PX * TILE_PX
        floor, w, sprite = t["floor"], t["w"], t["sprite"]
        loss = ((num.double() - (floor * w).sum()) / (w.sum() * px) * (IMG_H * IMG_W)).float()
        stats = {"recon_excess": ((tot.double() - floor.sum()) / 3.0 / (z.shape[0] * IMG_H * IMG_W)).float(),
                 "recon_sprite_excess": ((spr.double() - (floor * sprite).sum()) / 3.0
                                         / (sprite.sum() * px).clamp_min(1.0)).float()}
        return loss, stats

    def _chunk_recon(self, z, rgb, sprite, w):
        """One chunk of recon_loss -> (tile-weighted BCE sum, BCE sum, BCE sum on sprite tiles; the last two
        detached)."""
        logits = self.decoder(z)
        bce = _tile_sum(F.binary_cross_entropy_with_logits(logits.float(), rgb, reduction="none").sum(1))
        return (bce * w).sum(), bce.detach().sum(), (bce.detach() * sprite).sum()

    # dec_targets_once=False: targets and entropy floor recomputed inside every chunk
    def _recon_loss_legacy(self, z, obs, sprite_weight=5.0):
        ckpt = torch.is_grad_enabled() and z.requires_grad
        tot = None
        for i in range(0, z.shape[0], self.DEC_CHUNK):
            j = i + self.DEC_CHUNK
            args = (z[i:j], obs["screen"][i:j], obs["palette"][i:j], obs["blocks"][i:j], sprite_weight)
            run = self._chunk_ckpt if eager_only() else self._chunk_run or self._chunk_ckpt
            out = run(*args) if ckpt else self._chunk_recon_legacy(*args)
            tot = out if tot is None else tuple(a + b for a, b in zip(tot, out))
        num, w_sum, ex_sum, ex_sprite, n_sprite = tot
        loss = num / w_sum * (IMG_H * IMG_W)
        stats = {"recon_excess": ex_sum / (z.shape[0] * IMG_H * IMG_W),
                 "recon_sprite_excess": ex_sprite / n_sprite.clamp_min(1.0)}
        return loss, stats

    def _chunk_recon_legacy(self, z, screen, palette, blocks, sprite_weight):
        """One chunk of recon_loss with dec_targets_once=False -> (sum of sprite-weighted excess BCE, sum of the
        weights; detached stat sums)."""
        logits = self.decoder(z)
        with torch.no_grad():
            rgb = frame_rgb(screen, palette)
        bce = F.binary_cross_entropy_with_logits(logits.float(), rgb, reduction="none")
        excess = (bce - bce_floor(rgb)).sum(1)                                     # [n, H, W]
        sprite = sprite_pixel_mask(blocks).float()
        w = 1.0 + (sprite_weight - 1.0) * sprite
        ex = excess.detach() / 3.0                                                 # per colour value
        return (excess * w).sum(), w.sum(), ex.sum(), (ex * sprite).sum(), sprite.sum()


def _tile_sum(x):
    """[N, 96, 120] -> per game tile sums [N, 8, 10]."""
    return x.view(-1, BLOCK_ROWS, TILE_PX, BLOCK_COLS, TILE_PX).sum((2, 4))


# === Dynamics =======================================================================================================
# Dynamics: transformer state-space model (TSSM) with a KV ring cache, the prior head and a CUDA-graphed dream step.
#
# Positions: RoPE makes every query-key score depend on the relative offset only, but the attention sink's key is not
# rotated, so a query's sink score depends on its ABSOLUTE position. Training only ever sees positions 0 .. seq_len - 1,
# while acting counts up to the episode length. With `sink_max_pos` set (seq_len - 1), the sink key is rotated by
# pos - min(pos, sink_max_pos) at inference (acting, dreams, open-loop rollouts): the sink score of any query is then the
# one it had at a trained position, and every query-key score is unchanged.


# --- Primitives -----------------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """RMS normalisation computed in fp32."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight).to(dt)


def rope_cos_sin(pos, half_dim, base=10000.0):
    """RoPE cos / sin tables for positions `pos`."""
    inv_freq = base ** (-torch.arange(half_dim, device=pos.device, dtype=torch.float32) / half_dim)
    angles = pos.to(torch.float32).unsqueeze(-1) * inv_freq
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """Rotates the two halves of the last dim of x."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


# --- KV ring cache --------------------------------------------------------------------------------------------------
class TSSMCache:
    """Per-row ring of W key / value slots; updated only in place, so captured CUDA graphs keep working on it."""

    def __init__(self, n_layers, batch, window, n_kv_heads, head_dim, device, dtype=torch.float32):
        self.k = torch.zeros(n_layers, batch, window, n_kv_heads, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.key_pos = torch.full((batch, window), -1, dtype=torch.long, device=device)   # step held by a slot
        self.pos = torch.zeros(batch, dtype=torch.long, device=device)                    # position of next token
        self.window = window

    def reset_rows(self, rows):
        """Empties the given rows."""
        self.pos[rows] = 0
        self.key_pos[rows] = -1

    def snapshot(self):
        """Copy of the full cache state."""
        return [t.clone() for t in (self.k, self.v, self.key_pos, self.pos)]

    def restore(self, snap):
        """Restores a snapshot in place."""
        for t, s in zip((self.k, self.v, self.key_pos, self.pos), snap):
            t.copy_(s)


# --- Transformer block ----------------------------------------------------------------------------------------------
ATTN_GATES = ("none", "headwise", "elementwise")


class TSSMBlock(nn.Module):
    """Llama-style block: RMSNorm, QK-norm, RoPE, GQA with a learned sink, optional gated attention, SwiGLU."""

    def __init__(self, d_model, n_heads, n_kv_heads, ffn_hidden, attn_gate="elementwise"):
        super().__init__()
        assert attn_gate in ATTN_GATES, f"attn_gate must be one of {ATTN_GATES}"
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = hd = d_model // n_heads
        self.groups = n_heads // n_kv_heads
        self.norm_attn, self.norm_mlp = RMSNorm(d_model), RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, n_heads * hd, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(n_heads * hd, d_model, bias=False)
        self.q_norm, self.k_norm = RMSNorm(hd), RMSNorm(hd)
        self.sink_k = nn.Parameter(torch.randn(n_kv_heads, hd) * 0.02)          # always-visible attention sink
        self.sink_v = nn.Parameter(torch.zeros(n_kv_heads, hd))
        self.gate_proj = nn.Linear(d_model, ffn_hidden, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_hidden, bias=False)
        self.down_proj = nn.Linear(ffn_hidden, d_model, bias=False)
        self.gate_kind = attn_gate
        if attn_gate != "none":                                                  # gate = 2 sigmoid(Wx), 1 at init
            self.attn_gate = nn.Linear(d_model, n_heads * (hd if attn_gate == "elementwise" else 1), bias=False)
            nn.init.zeros_(self.attn_gate.weight)

    def _mlp(self, x):
        """SwiGLU feed-forward."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def _project(self, x, cos, sin):
        """-> RoPE-rotated q, k, v [..., heads, head_dim] and the attention gate (None without gating)."""
        h = self.norm_attn(x)
        shape = h.shape[:-1]
        q = self.q_proj(h).view(*shape, self.n_heads, self.head_dim)
        k = self.k_proj(h).view(*shape, self.n_kv_heads, self.head_dim)
        v = self.v_proj(h).view(*shape, self.n_kv_heads, self.head_dim)
        q = apply_rope(self.q_norm(q), cos, sin).to(v.dtype)
        k = apply_rope(self.k_norm(k), cos, sin).to(v.dtype)
        g = None
        if self.gate_kind != "none":
            g = (2.0 * torch.sigmoid(self.attn_gate(h).float())).view(*shape, self.n_heads, -1)
        return q, k, v, g

    def _gated(self, attn, g):
        """Multiplies the per-head attention output by the gate."""
        if g is None:
            return attn
        shape = attn.shape[:-1]
        return (attn.view(*shape, self.n_heads, self.head_dim) * g.to(attn.dtype)).reshape(*shape, -1)

    def _with_sink(self, keys, values):
        """Prepends the sink to keys / values [N, L, Hkv, hd]."""
        n = keys.shape[0]
        sk = self.k_norm(self.sink_k).to(keys.dtype).expand(n, 1, -1, -1)
        sv = self.sink_v.to(values.dtype).expand(n, 1, -1, -1)
        return torch.cat([sk, keys], dim=1), torch.cat([sv, values], dim=1)

    def _sink_scores(self, qg, sink_rot=None):
        """Returns the scores [N, Hkv, G] of grouped queries qg [N, Hkv, G, hd] on the sink key; sink_rot: per-row
        RoPE tables (cos, sin) [N, 1, hd / 2] that rotate the sink key first (TSSM.sink_rot), None: unrotated."""
        sk = self.k_norm(self.sink_k).to(qg.dtype)                               # [Hkv, hd]
        if sink_rot is None:
            return torch.einsum("nkgd,kd->nkg", qg, sk)
        skr = apply_rope(sk.expand(qg.shape[0], -1, -1), *sink_rot).to(qg.dtype)  # [N, Hkv, hd]
        return torch.einsum("nkgd,nkd->nkg", qg, skr)

    def _attend(self, q, keys, values, key_mask, sink_rot=None):
        """Single-token attention of q [N, H, hd] over sink + keys [N, L, Hkv, hd] with key_mask [N, L]. The sink's
        score and value are computed separately, so the cache is read in place and never copied."""
        N = q.shape[0]
        qg = q.view(N, self.n_kv_heads, self.groups, self.head_dim)
        sv = self.sink_v.to(q.dtype)
        key_mask = torch.cat([key_mask.new_ones(N, 1), key_mask], dim=1)
        scores = torch.cat([self._sink_scores(qg, sink_rot)[..., None],
                            torch.einsum("nkgd,nlkd->nkgl", qg, keys.to(q.dtype))], dim=-1).float()
        scores = (scores / math.sqrt(self.head_dim)).masked_fill(~key_mask[:, None, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1).to(q.dtype)
        out = probs[..., :1] * sv[None, :, None, :] + torch.einsum("nkgl,nlkd->nkgd", probs[..., 1:],
                                                                   values.to(q.dtype))
        return out.reshape(N, -1)

    def forward_train(self, x, cos, sin, attn_mask):
        """Teacher-forced pass over x [B, T, D] -> (x, (k, v)) with k, v [B, T, Hkv, hd]."""
        B, T, _ = x.shape
        q, k, v, g = self._project(x, cos, sin)
        keys, values = self._with_sink(k, v)
        if self.groups > 1:
            keys = keys.repeat_interleave(self.groups, dim=2)
            values = values.repeat_interleave(self.groups, dim=2)
        attn = F.scaled_dot_product_attention(q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2),
                                              attn_mask=attn_mask)
        x = x + self.o_proj(self._gated(attn.transpose(1, 2).reshape(B, T, -1), g))
        return x + self._mlp(self.norm_mlp(x)), (k, v)

    def forward_step(self, x, k_cache, v_cache, slot, key_mask, cos, sin, sink_rot=None):
        """Inference step: writes this token's K/V into ring slot `slot`, then attends to the cache."""
        q, k, v, g = self._project(x, cos, sin)
        rows = torch.arange(x.shape[0], device=x.device)
        k_cache[rows, slot] = k.to(k_cache.dtype)
        v_cache[rows, slot] = v.to(v_cache.dtype)
        x = x + self.o_proj(self._gated(self._attend(q, k_cache, v_cache, key_mask, sink_rot), g))
        return x + self._mlp(self.norm_mlp(x))

    def prefix_kv(self, keys, values):
        """Fixed open-loop context: sink + keys / values [N, L, Hkv, hd] -> [N, Hkv, 1 + L, hd]."""
        keys, values = self._with_sink(keys, values)
        return keys.transpose(1, 2).contiguous(), values.transpose(1, 2).contiguous()

    def forward_append(self, x, prefix_k, prefix_v, prefix_mask, new_k, new_v, new_mask, cos, sin, sink_rot=None):
        """Differentiable open-loop step over the fixed prefix [N, Hkv, Lp, hd] (sink included), the tokens appended
        so far and itself; the prefix is shared, never copied. -> (x, new_k + [k], new_v + [v])."""
        q, k, v, g = self._project(x, cos, sin)
        new_k, new_v = new_k + [k], new_v + [v]
        N = x.shape[0]
        qg = q.view(N, self.n_kv_heads, self.groups, self.head_dim)
        kn = torch.stack(new_k, dim=2).to(q.dtype)                              # [N, Hkv, j+1, hd]
        vn = torch.stack(new_v, dim=2).to(q.dtype)
        pre = torch.matmul(qg, prefix_k.to(q.dtype).transpose(-1, -2))
        if sink_rot is not None:                                                 # prefix column 0 is the sink
            pre = torch.cat([self._sink_scores(qg, sink_rot)[..., None], pre[..., 1:]], dim=-1)
        scores = torch.cat([pre, torch.matmul(qg, kn.transpose(-1, -2))], dim=-1).float() / math.sqrt(self.head_dim)
        mask = torch.cat([prefix_mask, new_mask], dim=1)
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        Lp = prefix_k.shape[2]
        attn = torch.matmul(probs[..., :Lp], prefix_v.to(v.dtype)) + torch.matmul(probs[..., Lp:], vn.to(v.dtype))
        x = x + self.o_proj(self._gated(attn.reshape(N, -1), g))
        return x + self._mlp(self.norm_mlp(x)), new_k, new_v


# --- TSSM -----------------------------------------------------------------------------------------------------------
class TSSM(nn.Module):
    """One token per step (z_t, a_t) -> h_{t+1}; causal sliding window of `window` steps plus a sink."""

    def __init__(self, latent_dim=1600, action_dim=12, d_model=1024, n_layers=6, n_heads=4, n_kv_heads=4,
                 ffn_hidden=2730, window=32, embed_hidden=None, attn_gate="elementwise", sink_max_pos=None):
        super().__init__()
        self.latent_dim, self.action_dim = latent_dim, action_dim
        self.d_model, self.n_kv_heads = d_model, n_kv_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.sink_max_pos = None if sink_max_pos is None else int(sink_max_pos)   # last trained position
        eh = embed_hidden or ffn_hidden
        self.token_mlp = nn.Sequential(nn.Linear(latent_dim + action_dim, eh), nn.SiLU(), nn.Linear(eh, d_model))
        self.embed_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([TSSMBlock(d_model, n_heads, n_kv_heads, ffn_hidden, attn_gate)
                                     for _ in range(n_layers)])
        self.norm_out = RMSNorm(d_model)

    def embed(self, z, action):
        """Token u_t = RMSNorm(MLP([z_t, a_t]))."""
        return self.embed_norm(self.token_mlp(torch.cat([z, action.to(z.dtype)], dim=-1)))

    def _rope(self, pos):
        """RoPE tables for positions `pos`."""
        return rope_cos_sin(pos, self.head_dim // 2)

    def sink_rot(self, pos):
        """Returns per-row RoPE tables (cos, sin) [N, 1, hd / 2] that rotate the sink key by pos - min(pos,
        sink_max_pos), so a query at any position gets the sink score of a trained position (None when off).
        Tensor ops only (CUDA-graph safe); identity rotation up to sink_max_pos."""
        if self.sink_max_pos is None:
            return None
        cos, sin = self._rope(pos - pos.clamp(max=self.sink_max_pos))
        return cos[:, None], sin[:, None]

    def _key_mask(self, key_pos, pos):
        """Keys that are filled and inside the window of each row."""
        return (key_pos >= 0) & (key_pos > (pos - self.window)[:, None])

    # teacher-forced pass
    def forward_sequence(self, z, actions, return_kv=False, detach_kv=True):
        """z [B, T, D], actions [B, T, A] -> (h [B, T, d] with h[:, t] = h_{t+1}, per-layer (k, v) or None)."""
        x = self.embed(z, actions)
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        cos, sin = self._rope(pos)
        cos, sin = cos[None, :, None], sin[None, :, None]
        rel = pos[:, None] - pos[None, :]
        causal = (rel >= 0) & (rel < self.window)
        attn_mask = torch.cat([causal.new_ones(T, 1), causal], dim=1)            # column 0 = the sink
        kv = []
        eager = eager_only()                             # bypass a compiled blk.forward_train
        for blk in self.blocks:
            x, (k, v) = (TSSMBlock.forward_train(blk, x, cos, sin, attn_mask) if eager
                         else blk.forward_train(x, cos, sin, attn_mask))
            if return_kv:
                kv.append((k.detach(), v.detach()) if detach_kv else (k, v))
        return self.norm_out(x), (kv if return_kv else None)

    # inference on the ring cache
    def make_cache(self, batch, device, dtype=torch.float32):
        """Empty ring cache for `batch` rows."""
        return TSSMCache(len(self.blocks), batch, self.window, self.n_kv_heads, self.head_dim, device, dtype)

    @torch.no_grad()
    def prefill(self, cache, kv, rows, t0):
        """Loads teacher-forced K/V of sequences `rows`, tokens t0-W+1 .. t0, into the ring (no host sync)."""
        N, W = rows.shape[0], cache.window
        dev = rows.device
        steps = t0[:, None] - (W - 1) + torch.arange(W, device=dev)[None]        # [N, W], oldest first
        src = steps.clamp(min=0)
        slot = steps % W
        r = torch.arange(N, device=dev)[:, None].expand(N, W)
        for li, (k, v) in enumerate(kv):
            cache.k[li][r, slot] = k[rows[:, None], src].to(cache.k.dtype)
            cache.v[li][r, slot] = v[rows[:, None], src].to(cache.v.dtype)
        cache.key_pos[r, slot] = torch.where(steps >= 0, steps, torch.full_like(steps, -1))
        cache.pos.copy_(t0 + 1)

    def forward_step(self, z, action, cache):
        """One token per row on the cache (acting, dreams) -> h [N, d]; the cache advances in place."""
        x = self.embed(z, action)
        pos = cache.pos
        slot = pos % cache.window
        rows = torch.arange(x.shape[0], device=x.device)
        cache.key_pos[rows, slot] = pos
        cos, sin = self._rope(pos)
        cos, sin = cos[:, None], sin[:, None]
        key_mask = self._key_mask(cache.key_pos, pos)
        sink_rot = self.sink_rot(pos)
        for li, blk in enumerate(self.blocks):
            x = blk.forward_step(x, cache.k[li], cache.v[li], slot, key_mask, cos, sin, sink_rot)
        cache.pos.add_(1)
        return self.norm_out(x)

    # differentiable open-loop rollouts
    def open_loop_state(self, kv, rows, prefix_len):
        """Rollout context: K/V of sequences `rows` with the first `prefix_len` tokens visible (+ sink), built once
        and shared by every open-loop step."""
        T = kv[0][0].shape[1]
        steps = torch.arange(T, device=rows.device)[None]
        key_pos = torch.where(steps < prefix_len[:, None], steps, torch.full_like(steps, -1))
        prefix = [blk.prefix_kv(k[rows], v[rows]) for blk, (k, v) in zip(self.blocks, kv)]
        return {"prefix": prefix, "new_k": [[] for _ in self.blocks], "new_v": [[] for _ in self.blocks],
                "key_pos": key_pos, "new_pos": prefix_len.new_zeros(prefix_len.shape[0], 0),
                "pos": prefix_len.clone()}

    def step_append(self, z, action, state):
        """One differentiable token per row on an open-loop state -> h [N, d]."""
        x = self.embed(z, action)
        pos = state["pos"]
        cos, sin = self._rope(pos)
        cos, sin = cos[:, None], sin[:, None]
        key_mask = self._key_mask(state["key_pos"], pos)
        prefix_mask = torch.cat([key_mask.new_ones(x.shape[0], 1), key_mask], dim=1)        # column 0 = the sink
        new_pos = torch.cat([state["new_pos"], pos[:, None]], dim=1)
        new_mask = self._key_mask(new_pos, pos)
        sink_rot = self.sink_rot(pos)
        for li, blk in enumerate(self.blocks):
            pk, pv = state["prefix"][li]
            x, state["new_k"][li], state["new_v"][li] = blk.forward_append(
                x, pk, pv, prefix_mask, state["new_k"][li], state["new_v"][li], new_mask, cos, sin, sink_rot)
        state["new_pos"] = new_pos
        state["pos"] = pos + 1
        return self.norm_out(x)


# --- Prior head -----------------------------------------------------------------------------------------------------
class PriorHead(nn.Module):
    """MLP h_{t+1} -> logits [..., L, K] of the prior p(z_{t+1} | h_{t+1})."""

    def __init__(self, in_dim, n_latents=40, classes=40, hidden=1024, layers=3, unimix=0.01):
        super().__init__()
        self.n_latents, self.classes, self.unimix = n_latents, classes, unimix
        mods, d = [], in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
            d = hidden
        mods.append(nn.Linear(d, n_latents * classes))
        self.net = nn.Sequential(*mods)

    def forward(self, h):
        return self.net(h).float().reshape(*h.shape[:-1], self.n_latents, self.classes)

    def logprobs(self, h, unimix=None):
        """Log-probabilities with the uniform mix (`unimix` overrides it)."""
        return unimix_logprobs(self(h), self.unimix if unimix is None else unimix)


# --- CUDA-graphed dream step ----------------------------------------------------------------------------------------
class GraphedDreamStep:
    """One imagined step (TSSM step + prior head) for n dreams, captured as a CUDA graph and checked vs eager."""

    def __init__(self, tssm, prior, n, cache_dtype=torch.float32, amp_dtype=None, device="cuda", capture=True,
                 warmup=3):
        self.tssm, self.prior, self.n = tssm, prior, n
        self.L, self.K = prior.n_latents, prior.classes
        self.amp_dtype = amp_dtype
        self.device = torch.device(device)
        self.cache = tssm.make_cache(n, self.device, cache_dtype)               # static cache read by the graph
        self.action = torch.zeros(n, tssm.action_dim, device=self.device)
        self.z_prev = torch.zeros(n, tssm.latent_dim, device=self.device)
        self.graph = None
        if capture and self.device.type == "cuda":
            self._capture(warmup)

    def _body(self):
        """Graph body; autocast's weight cache stays off so replays read the current weights; compiled modules run
        eagerly here (the graph replays the captured kernels)."""
        with torch.no_grad(), eager_modules(), torch.autocast(self.device.type, dtype=self.amp_dtype or torch.bfloat16,
                                                              enabled=self.amp_dtype is not None, cache_enabled=False):
            h = self.tssm.forward_step(self.z_prev, self.action, self.cache).float()
            logits = self.prior(h)
        return h, logits

    def _randomize(self):
        """Random one-hot inputs for warm-up and the capture check."""
        self.action.copy_(F.one_hot(torch.randint(0, self.action.shape[1], (self.n,), device=self.device),
                                    self.action.shape[1]).float())
        self.z_prev.copy_(F.one_hot(torch.randint(0, self.K, (self.n, self.L), device=self.device),
                                    self.K).float().flatten(1))

    def _capture(self, warmup):
        """Warms up, captures the graph and checks it against the eager step."""
        snap = self.cache.snapshot()
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                self._randomize()
                self._body()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.cache.restore(snap)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            self.h_out, self.logits_out = self._body()
        self.graph = graph

        self._randomize()
        self.cache.restore(snap)
        self.graph.replay()
        h_g, l_g = self.h_out.clone(), self.logits_out.clone()
        self.cache.restore(snap)
        h_e, l_e = self._body()
        self.cache.restore(snap)
        err = max((h_g - h_e).abs().max().item(), (l_g - l_e).abs().max().item())
        tol = 5e-2 if self.amp_dtype is not None else 1e-3
        if not err <= tol:
            self.graph = None
            raise RuntimeError(f"graph replay differs from the eager step by {err:.3g} (tolerance {tol})")

    def __call__(self, action, z_prev, unimix=0.0, gumbel=None):
        """action [n, A], z_prev [n, L*K] -> (h [n, d], one-hot sample z [n, L*K], prior log-probs [n, L, K])."""
        self.action.copy_(action)
        self.z_prev.copy_(z_prev)
        if self.graph is None:
            h, logits = self._body()
        else:
            self.graph.replay()
            h, logits = self.h_out.clone(), self.logits_out.clone()
        logp = unimix_logprobs(logits, unimix)
        g = gumbel_like(logp) if gumbel is None else gumbel
        z = F.one_hot((logp + g).argmax(-1), self.K).float().flatten(1)
        return h, z, logp
