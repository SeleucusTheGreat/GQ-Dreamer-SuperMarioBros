"""GQ-Dreamer agent: world-model training, imagination (actor-critic), acting in the game, play rounds and the dream
audit.

The networks it builds are in model.py; this file runs and trains them.
"""
import contextlib
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .audit import AUDIT_KINDS, SnapshotPool, audit_enabled, audit_metrics, rmse, select_starts
from .checkpoint import CheckpointMixin
from .config import GQ_CONFIG, config_to_dict
from .game import ACTION_LEFT, ACTION_RIGHT, GRID_CENTER_INDEX, GRID_DIM, MAIN_LEVELS, MAP_NAMES
from .model import (TSSM, Actor, ContinueHead, Critic, GraphedDreamStep, GridHead, PixelDecoder, PriorHead, Tokenizer,
                    TwoHotEncoding, TwoHotHead, class_usage, entropy_categorical, gumbel_like, kl_categorical,
                    sample_onehot, unimix_logprobs)
from .observations import (IMG_H, IMG_W, frame_rgb, n_patches, obs_ram, obs_to_tensors, ram_fields, sample_hide,
                           sprite_pixel_mask, unpack_bits)
from .replay import BatchPrefetcher, ReplayBuffer, with_batch_extras
from .train_utils import (DevicePercentileClipper, LossHarmonizer, PercentileClipper, PhaseTimer, reinforce_loss,
                          actor_metrics, compile_available, compile_module, eager_modules, explained_variance,
                          lambda_returns, make_grad_scaler, masked_mean, resolve_amp_dtype, setup_compile,
                          weighted_moments)

torch.set_float32_matmul_precision("high")

ADV_STD_MIN = 1e-3                                                         # floor of each advantage stream's std


class GQDreamer(CheckpointMixin):
    """GroundQuery-Dreamer: query-fusion tokenizer + TSSM world model, actor-critic trained in imagination."""
    OBS_KEYS = ("screen", "palette", "grid", "map_id", "ram")              # what the encoder reads
    CURIOSITY_POS_WEIGHT = 5.0                                             # weight of non-zero curiosity targets
    DEATH_PROBE_STEPS = 4                                                  # open-loop steps of the dream death probe
    DEATH_PROBE_EVERY = 5                                                  # probe every n-th update (round average)
    # world-model loss terms, each multiplied by its fixed weight (they sum to world_model_loss); round averages
    WM_LOSS_TERMS = ("pixel", "level", "status", "kl_dyn", "kl_rep", "rollout", "sparse_reward", "standard_reward",
                     "continue", "grid", "curiosity")
    STREAMS = ("ext", "cur")                                               # advantage streams: extrinsic, curiosity

    def __init__(self, cfg=GQ_CONFIG, device="cuda", envs=None, verbose=True):
        c = self.cfg = cfg
        assert c.seq_len > c.context_length, "training sequences must be longer than the context window"
        assert c.rollout_steps == 0 or c.seq_len >= c.rollout_steps + 2, "the rollout needs rollout_steps + 2 frames"
        self.device = torch.device(device)
        self.envs = envs
        self.verbose = verbose
        self.action_dim = c.action_dim
        self.n_latents, self.classes = c.n_latents, c.classes
        self.latent_dim = c.n_latents * c.classes
        self.tssm_dim = c.tssm_dim
        self.state_dim = c.tssm_dim + self.latent_dim                      # s = [h, flat one-hot z]

        # run state
        self.total_num_episodes = self.total_num_steps = self.total_num_updates = 0
        self.last_play_stats = {}
        self.last_drift = float("nan")
        self._probe = self._probe_ref = None
        self._prefetcher = None
        self._death_probe_acc = None                                       # round sums of the death probe
        self._wm_loss_acc = None                                           # round sums of the weighted loss terms
        self._dream_graph = None
        self._use_dream_graph = c.dream_cuda_graph
        self._act_graphs = {}                                              # n envs -> GraphedActStep
        self._use_act_graph = c.act_cuda_graph
        self._pending_prio = None                                          # surprise priorities of the last update
        self._resumed = False
        self.last_grad_shares = {}                                         # newest per-term gradient norms / shares
        self._grad_share_ok = True
        self.audit_snapshots = SnapshotPool(c.audit_snapshot_every if audit_enabled(c) else 0)   # audit starts
        self._audit_env = None                                             # the audit's own emulator (first audit)
        self.timer = PhaseTimer(self.device)
        # 255-bin targets of the reward / curiosity heads and the critics (HL-Gauss or two-hot)
        self.two_hot = TwoHotEncoding(device=self.device, target=c.value_target, sigma_ratio=c.hl_sigma_ratio)
        if c.adv_norm != "std":
            raise ValueError(f"adv_norm must be \"std\" (per-stream std; the other modes were removed), not {c.adv_norm!r}")
        if c.death_penalty and c.level_end_terminal:
            print("\033[93m    [setup] death_penalty with level_end_terminal=True: the continue head also learns level "
                  "completions, so the dreams would charge them as deaths too\033[0m")

        self._build_networks()
        self._build_optimizers()
        self._compile_networks()
        self._set_file_names()
        num_envs = len(envs) if envs is not None else 1
        self.buffer = ReplayBuffer(
            capacity=c.buffer_size, num_envs=num_envs, frames_per_step=c.frames_per_step,
            death_sample_prob=c.death_sample_prob, recent_sample_prob=c.recent_sample_prob,
            reward_sample_prob=c.reward_sample_prob, reward_threshold=c.reward_threshold,
            loss_sample_prob=c.loss_sample_prob, loss_priority_alpha=c.loss_priority_alpha)
        self._print_setup()

    # --- Setup ------------------------------------------------------------------------------------------------------
    def _build_networks(self):
        """World model (tokenizer, TSSM, prior, heads) and behaviour (actor, critics, EMA copies)."""
        c, dev, S = self.cfg, self.device, self.state_dim
        self.tokenizer = Tokenizer(c.n_latents, c.classes, c.enc_dim, c.enc_depth, c.enc_heads,
                                   dec_chans=tuple(c.dec_chans), head_hidden=c.head_hidden, unimix=c.unimix,
                                   grad_checkpoint=c.enc_grad_checkpoint, enc_stem=c.enc_stem,
                                   enc_patch_tiles=c.enc_patch_tiles, enc_native=c.enc_input == "native",
                                   decoder=c.decoder, dec_dim=c.dec_dim, dec_depth=c.dec_depth,
                                   dec_heads=c.dec_heads, dec_patch_tiles=c.dec_patch_tiles,
                                   enc_pos_inject=c.enc_pos_inject, enc_swiglu=c.enc_swiglu,
                                   enc_ffn=c.enc_ffn, enc_fast_attn=c.enc_fast_attn, dec_subpixel=c.dec_subpixel,
                                   dec_sac=c.dec_sac, dec_targets_once=c.dec_targets_once,
                                   enc_last_queries=c.enc_last_queries).to(dev)
        # sink_max_pos: training sequences hold positions 0 .. seq_len - 1; acting counts higher
        self.tssm = TSSM(self.latent_dim, c.action_dim, c.tssm_dim, c.tssm_layers, c.tssm_heads, c.tssm_kv_heads,
                         c.tssm_ffn, c.context_length, attn_gate=c.attn_gate,
                         sink_max_pos=c.seq_len - 1).to(dev)
        self.prior = PriorHead(c.tssm_dim, c.n_latents, c.classes, hidden=c.mlp_dim, layers=c.prior_layers,
                               unimix=c.unimix).to(dev)
        self.sparse_reward_head = TwoHotHead(S, hidden=c.mlp_dim).to(dev)
        self.standard_reward_head = TwoHotHead(S, hidden=c.mlp_dim).to(dev)
        self.curiosity_head = TwoHotHead(S, hidden=c.mlp_dim, zero_bias_bin=127).to(dev)
        self.continue_head = ContinueHead(S, hidden=c.mlp_dim).to(dev)
        self.grid_head = GridHead(S, hidden=c.mlp_dim).to(dev)            # explored 5x5 grid from s = [h, z]

        self.actor = Actor(c.action_dim, S, c.mlp_dim).to(dev)
        self.critic = Critic(S, mlp_dim=c.mlp_dim).to(dev)
        self.curiosity_critic = Critic(S, mlp_dim=c.mlp_dim).to(dev)
        self.critic_target = copy.deepcopy(self.critic)
        self.curiosity_critic_target = copy.deepcopy(self.curiosity_critic)
        for m in (self.critic_target, self.curiosity_critic_target):
            m.requires_grad_(False)
        self.harmony = LossHarmonizer().to(dev)                            # always built: one checkpoint layout

    def _build_optimizers(self):
        """One AdamW for the whole world model, Adam for actor, critics and loss weights."""
        c = self.cfg
        fused = {"fused": True} if c.sync_free and self.device.type == "cuda" else {}   # steps skipped on the GPU
        self.wm_params = self.optimizer_params("wm_optimizer")
        self.wm_optimizer = torch.optim.AdamW(self.wm_params, lr=c.wm_lr, weight_decay=0.01, **fused)
        clipper = DevicePercentileClipper if c.sync_free else PercentileClipper
        self.wm_clipper = clipper(c.clip_percentile, c.clip_window, c.clip_warmup, c.clip_initial,
                                  **({"device": self.device} if c.sync_free else {}))
        self.actor_optimizer = torch.optim.Adam(self.optimizer_params("actor_optimizer"), lr=c.actor_lr, **fused)
        self.critic_optimizer = torch.optim.Adam(self.optimizer_params("critic_optimizer"), lr=c.critic_lr, **fused)
        self.curiosity_critic_optimizer = torch.optim.Adam(self.optimizer_params("curiosity_critic_optimizer"),
                                                           lr=c.critic_lr, **fused)
        self.harmony_optimizer = torch.optim.Adam(self.optimizer_params("harmony_optimizer"), lr=c.harmony_lr)
        self.amp_dtype = resolve_amp_dtype(self.device, c.mixed_precision)
        self.cache_dtype = self.amp_dtype or torch.float32                 # KV caches of acting / dreams
        self.scalers = {name: make_grad_scaler(self.device, self.amp_dtype == torch.float16)
                        for name in self.OPTIMIZERS}

    def _compile_networks(self):
        """cfg.compile: torch.compile of the encoder's checkpointed layers, the decoder's checkpointed chunks, the MLP
        heads / actor / critics / prior and the TSSM's teacher-forced blocks. Parameters and state_dict keys are
        unchanged; the hand-captured CUDA graphs (dreams, acting) run the eager modules. The first updates compile
        (minutes; cached on disk afterwards)."""
        self.compiled = False
        if not self.cfg.compile:
            return
        ok, why = compile_available(self.device)
        if not ok:
            print(f"\033[93m    [setup] compile=True but {why}: running eagerly\033[0m")
            return
        setup_compile()
        self.tokenizer.encoder.compile_layers()
        if isinstance(self.tokenizer.decoder, PixelDecoder):
            self.tokenizer.compile_decoder()
        for name in ("actor", "critic", "curiosity_critic", "critic_target", "curiosity_critic_target",
                     "sparse_reward_head", "standard_reward_head", "curiosity_head", "continue_head", "grid_head",
                     "prior"):
            compile_module(getattr(self, name).net)
        for blk in self.tssm.blocks:
            blk.forward_train = torch.compile(blk.forward_train)
        self.compiled = True

    def _print_setup(self):
        """One-line model summary."""
        c = self.cfg

        def count(*mods):
            return sum(p.numel() for m in mods for p in m.parameters()) / 1e6
        heads = count(self.sparse_reward_head, self.standard_reward_head, self.curiosity_head, self.continue_head,
                      self.grid_head)
        amp = str(self.amp_dtype).replace("torch.", "") if self.amp_dtype else "off (fp32)"
        dec = f"{c.decoder} decoder" + (f" ({n_patches(c.dec_patch_tiles)} queries)" if c.decoder == "transformer"
                                        else "")
        print(f"\033[90m    [setup] GQ-Dreamer {self.ARCH_VERSION} \"{c.config_name}\" | encoder "
              f"{count(self.tokenizer.encoder):.1f}M, decoder {count(self.tokenizer.decoder):.1f}M, grounding heads "
              f"on z {count(self.tokenizer.heads):.2f}M | TSSM {count(self.tssm):.1f}M | prior "
              f"{count(self.prior):.1f}M | heads (incl. grid on [h, z]) {heads:.1f}M | input "
              f"{'128x160' if self.tokenizer.encoder.native else f'{IMG_H}x{IMG_W}'} -> "
              f"{self.tokenizer.encoder.n_image_tokens} image tokens ({c.enc_stem} stem), encoder {c.enc_dim} x "
              f"{c.enc_depth}{f' SwiGLU {c.enc_ffn}' if c.enc_swiglu else ''}"
              f"{' (2D position in every layer)' if c.enc_pos_inject else ''} | {dec} -> "
              f"{IMG_H}x{IMG_W} | harmony {'on' if c.harmony else 'off'} | z = "
              f"{c.n_latents} one-hot x {c.classes} classes | s = [h({c.tssm_dim}), z({self.latent_dim})] | batch "
              f"{c.batch_size} x {c.seq_len}, {c.recon_frames or 'all'} frames decoded per update | "
              f"mixed precision: {amp}\033[0m")
        on = [k for k in ("compile", "act_cuda_graph", "dec_subpixel", "dec_sac", "dec_targets_once", "enc_fast_attn",
                          "skip_pad_frames", "sync_free", "h2d_stream", "async_save", "async_dream_pdf")
              if getattr(c, k) and (k != "compile" or self.compiled)]
        print(f"\033[90m    [setup] speed: {', '.join(on) or 'none'}"
              f"{' (the first updates compile: minutes, cached on disk afterwards)' if self.compiled else ''}\033[0m")

    @property
    def config(self):
        """Config as a plain dict (stored in checkpoints)."""
        return {**config_to_dict(self.cfg), "arch_version": self.ARCH_VERSION}

    # --- Batches ----------------------------------------------------------------------------------------------------
    def start_prefetch(self, batch_size=None, seq_len=None):
        """Samples batches on a background thread while the GPU trains (the buffer must not change meanwhile)."""
        self.stop_prefetch()
        cuda = self.device.type == "cuda"
        self._prefetcher = BatchPrefetcher(self.buffer, batch_size or self.cfg.batch_size,
                                           seq_len or self.cfg.seq_len, pin=cuda,
                                           device=self.device if cuda and self.cfg.h2d_stream else None)

    def stop_prefetch(self):
        """Stops the background sampler."""
        if self._prefetcher is not None:
            self._prefetcher.close()
            self._prefetcher = None

    def sample_batch(self, batch_size=None, seq_len=None):
        """Replay batch on device, decoded into model inputs and targets; None if the buffer is empty."""
        batch_size, seq_len = batch_size or self.cfg.batch_size, seq_len or self.cfg.seq_len
        pf = self._prefetcher
        use_pf = pf is not None and (pf.batch_size, pf.seq_len) == (batch_size, seq_len)
        item = pf.get() if use_pf else self.buffer.sample(batch_size, seq_len)
        if item is None:
            return None
        if "__gpu__" in item:                                              # copied on the prefetcher's stream
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(item["__event__"])
            b, extra = item["__gpu__"], item["__cpu__"]
            for v in b.values():
                v.record_stream(stream)
        else:
            if "__cpu__" in item:
                extra, cpu = item["__cpu__"], item["__batch__"]
            else:                                                          # sampled here
                cpu, extra = with_batch_extras(item)
            b = {k: v.to(self.device, non_blocking=True) for k, v in cpu.items()}
        return {
            "screen": b["screens"],
            "palette": b["palettes"],
            "blocks": b["blocks"].long(),
            "grid": unpack_bits(b["grids"], GRID_DIM),
            "map_id": b["map_ids"].long(),
            "ram": b["rams"],
            "actions": F.one_hot(b["actions"].long(), self.action_dim).float(),
            "sparse_rewards": b["sparse_rewards"],
            "standard_rewards": b["standard_rewards"],
            "curiosities": b["curiosities"],
            "continues": b["continues"][..., 0],
            "valid": b["valid"],
            "episode_end": b["episode_end"],
            "index": b["index"],                                           # buffer rows (surprise priorities)
            "real_idx": b["real_idx"], "real_gather": b["real_gather"],    # real frames / position -> real frame
            **extra,                                                       # CPU: n_valid1, index_cpu, valid_cpu
        }

    @classmethod
    def _flat_obs(cls, batch, idx=None):
        """Encoder inputs + sprite overlay of a batch, flattened over [B, T] (optionally indexed)."""
        out = {}
        for k in cls.OBS_KEYS + ("blocks",):
            v = batch[k].flatten(0, 1)
            out[k] = v if idx is None else v[idx]
        return out

    # --- Small helpers ----------------------------------------------------------------------------------------------
    def _amp(self):
        """Autocast context (no-op when mixed precision is off)."""
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype or torch.bfloat16,
                              enabled=self.amp_dtype is not None)

    def _optimize(self, loss, name, params, max_norm=None):
        """Backward + optional clipping + step of optimizer `name` through its GradScaler -> pre-clip norm."""
        optimizer, scaler = getattr(self, name), self.scalers[name]
        if self.cfg.sync_free and not scaler.is_enabled() and optimizer.defaults.get("fused"):
            # no host sync: the threshold may be a GPU tensor, and the fused optimizer skips a non-finite step on the
            # GPU (found_inf, as GradScaler drives fused optimizers) -> pre-clip norm as a GPU tensor
            loss.backward()
            norm, optimizer.found_inf = None, None
            if max_norm is not None:
                grads = [p.grad for p in params if p.grad is not None]
                norm = (torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads))) if grads
                        else torch.zeros((), device=self.device))
                torch._foreach_mul_(grads, (max_norm / (norm + 1e-6)).clamp(max=1.0))
                optimizer.found_inf = (~torch.isfinite(norm)).float()
            optimizer.step()
            return norm
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = nn.utils.clip_grad_norm_(params, max_norm) if max_norm is not None else None
        if scaler.is_enabled():
            scaler.step(optimizer)                                         # skipped on fp16 overflow
            scaler.update()
        elif norm is None or torch.isfinite(norm):
            optimizer.step()
        return norm

    @torch.no_grad()
    def _critic_ema_update(self):
        """Moves the target critics towards the critics."""
        for tgt, src in ((self.critic_target, self.critic), (self.curiosity_critic_target, self.curiosity_critic)):
            for pt, po in zip(tgt.parameters(), src.parameters()):
                pt.lerp_(po, 1.0 - self.cfg.critic_ema_tau)

    def _defer_priorities(self, batch, dyn):
        """Surprise priorities without a host sync: this update's KL is copied to pinned memory asynchronously and
        written into the buffer at the next update (or flush_priorities), when the copy has long finished."""
        self.flush_priorities()
        val = torch.empty(dyn.shape, dtype=torch.float32, pin_memory=True)
        val.copy_(dyn.detach().float(), non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self._pending_prio = (ev, batch["index_cpu"][:, 1:], batch["valid_cpu"][:, 1:] > 0, val)

    def flush_priorities(self):
        """Writes the pending surprise priorities into the buffer (before it changes: end of every training round)."""
        if self._pending_prio is not None:
            ev, idx, sel, val = self._pending_prio
            ev.synchronize()
            self.buffer.update_priorities(idx[sel], val[sel])
            self._pending_prio = None

    @staticmethod
    def full_state(h, zf):
        """Agent state s = [h, z]."""
        return torch.cat([h.float(), zf.float()], dim=-1)

    def alive_prob(self, s, logit=None):
        """p(alive) of states s [..., S] from the continue head (or its `logit`); with cfg.continue_calibrated,
        log(continue_zero_weight) is added to the logit, which undoes the loss weighting of the cont = 0 frames (with
        weight w the head learns q = p / (p + w (1 - p)))."""
        logit = (self.continue_head(s) if logit is None else logit).float()
        w = float(self.cfg.continue_zero_weight)
        if self.cfg.continue_calibrated and w > 0 and w != 1.0:
            logit = logit + math.log(w)
        return torch.sigmoid(logit)

    def real_rewards(self, batch):
        """Real extrinsic rewards of the transitions of a batch, aligned with the states (entry j: state j = observation
        j + 1 -> observation j + 2) [B, T-1]: sparse + standard, plus cfg.death_penalty on every transition into a
        death row (cont = 0; a death only with level_end_terminal = False). Not stored in the buffer, so the penalty
        can change without relabelling it."""
        rows = (batch["sparse_rewards"] + batch["standard_rewards"])[..., 0].float()            # [B, T]: t -> t+1
        if self.cfg.death_penalty:
            nxt = (batch["continues"][:, 1:] <= 0).float() * batch["valid"][:, 1:]               # obs t+1 is a death
            rows = rows + torch.cat([self.cfg.death_penalty * nxt, torch.zeros_like(rows[:, :1])], dim=1)
        return rows[:, 1:]

    # --- Encoder ----------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_frames(self, obs, chunk=1024):
        """Observations [..., ...] -> (flat one-hot samples [..., D], posterior log-probs [..., L, K]), no gradient."""
        lead = obs["screen"].shape[:-2]
        flat = {k: obs[k].reshape(-1, *obs[k].shape[len(lead):]) for k in self.OBS_KEYS}
        n = flat["screen"].shape[0]
        zs, lps = [], []
        with self._amp():
            for i in range(0, n, chunk):
                z, lp = self.tokenizer.encode({k: v[i:i + chunk] for k, v in flat.items()})
                zs.append(z.float())
                lps.append(lp.float())
        return (torch.cat(zs).view(*lead, self.latent_dim),
                torch.cat(lps).view(*lead, self.n_latents, self.classes))

    @torch.no_grad()
    def _update_drift(self):
        """Every drift_interval updates: share of categoricals whose most likely class changed on fixed probe frames."""
        c = self.cfg
        if c.drift_interval <= 0 or self.total_num_updates % c.drift_interval:
            return
        if self._probe is None:
            n = min(c.drift_frames, len(self.buffer))
            if n < 2:
                return
            p = self.buffer.sample_frames(n)
            self._probe = {"screen": p["screens"].to(self.device), "palette": p["palettes"].to(self.device),
                           "grid": unpack_bits(p["grids"].to(self.device), GRID_DIM),
                           "map_id": p["map_ids"].long().to(self.device), "ram": p["rams"].to(self.device)}
        mode = self.encode_frames(self._probe)[1].argmax(-1)
        if self._probe_ref is not None:
            self.last_drift = (mode != self._probe_ref).float().mean().item()
        self._probe_ref = mode

    # --- World model: losses ----------------------------------------------------------------------------------------
    def _recon_branch(self, batch, z, dream):
        """Decoder + grounding heads (level / status), both on z only. Clean part: the posterior samples of
        recon_frames real frames (t >= 1; 0 = every valid frame). Augmented part: mask_frames of them re-encoded with
        MAE masking + token dropout, weighted mask_weight in the loss. Dream-like states only give grounding accuracies
        (no loss)."""
        c, dev = self.cfg, self.device
        T = z.shape[1]
        valid1 = batch["valid"][:, 1:].flatten()                           # frames t >= 1 (as the heads)
        n_valid = batch["n_valid1"] if "n_valid1" in batch else int(valid1.sum().item())   # CPU count: no sync
        if c.recon_frames <= 0 or c.recon_frames >= n_valid:
            sel1 = valid1.nonzero().squeeze(1)                             # index into [B, T - 1]
        else:
            sel1 = torch.multinomial(valid1, c.recon_frames, replacement=False)
        sel = (sel1 // (T - 1)) * T + sel1 % (T - 1) + 1                   # the same frames, index into [B, T]
        o = self._flat_obs(batch, sel)
        zc = z.flatten(0, 1)[sel]
        tgt = self.tokenizer.recon_targets(o, c.sprite_weight) if c.dec_targets_once else None   # once per update
        pix_loss, stats = self.tokenizer.recon_loss(zc, o, c.sprite_weight, targets=tgt)   # chunked + checkpointed
        g, acc = self.tokenizer.heads.losses(zc, o["map_id"], o["ram"])
        n_aug = min(c.mask_frames, sel.shape[0])
        hide, zero = None, torch.zeros((), device=dev)
        level_notoken = zero
        if n_aug > 0:                                                      # augmented re-encoding (MAE + dropout)
            pick = torch.randperm(sel.shape[0], device=dev)[:n_aug]
            oa = {k: v[pick] for k, v in o.items()}
            hide = sample_hide(oa["blocks"], c.mask_max, c.enc_patch_tiles)
            drop_p = torch.tensor((c.drop_grid, c.drop_level, c.drop_status), device=dev)
            drop = torch.rand(n_aug, 3, device=dev) < drop_p
            za = sample_onehot(self.tokenizer.posterior(oa, hide, drop)).flatten(1)
            pix_a, _ = self.tokenizer.recon_loss(za, oa, c.sprite_weight, targets=None if tgt is None
                                                 else self.tokenizer.subset_targets(tgt, pick))
            g_a, _ = self.tokenizer.heads.losses(za, oa["map_id"], oa["ram"])
            a = c.mask_weight
            pix_loss = (1.0 - a) * pix_loss + a * pix_a
            g = {k: (1.0 - a) * g[k] + a * g_a[k] for k in g}
            with torch.no_grad():
                no_level = drop[:, 1].float()
                lvl_hit = (self.tokenizer.heads.predict(za)[0] == oa["map_id"].long()).float()
                level_notoken = (lvl_hit * no_level).sum() / no_level.sum().clamp_min(1.0)
        parts = {"pixel": c.w_recon * pix_loss, "level": c.w_level * g["level"], "status": c.w_status * g["status"]}
        with torch.no_grad():
            tgt = dream["tgt"]                                             # dream-like states: accuracy only
            _, acc_d = self.tokenizer.heads.losses(dream["states"][..., self.tssm_dim:], tgt["map_id"], tgt["ram"],
                                                   mask=tgt["mask"])
            stats.update({"pix_loss": pix_loss.detach(),
                          "level_loss": g["level"].detach(), "status_loss": g["status"].detach(),
                          "level_acc": acc["level_acc"], "tier_acc": acc["tier_acc"],
                          "level_acc_notoken": level_notoken,
                          "grounding_loss": (parts["level"] + parts["status"]).detach(),
                          "dream_level_acc": acc_d["level_acc"], "dream_tier_acc": acc_d["tier_acc"],
                          "mask_ratio": hide.float().mean() if hide is not None else zero,
                          "recon_loss": (parts["pixel"] + parts["level"] + parts["status"]).detach()})
        return stats, parts

    @staticmethod
    def _targets(batch, rows, t):
        """Targets of the state at observation t: reward / curiosity of t-1 -> t, continue, grid, level and RAM of t."""
        v = batch["valid"][rows, t]
        return {"sparse": batch["sparse_rewards"][rows, t - 1, 0],
                "standard": batch["standard_rewards"][rows, t - 1, 0],
                "curiosity": batch["curiosities"][rows, t - 1, 0], "cont": batch["continues"][rows, t],
                "grid": batch["grid"][rows, t], "map_id": batch["map_id"][rows, t], "ram": batch["ram"][rows, t],
                "mask": v}

    def _head_losses(self, s, tgt):
        """Reward (sparse / standard), curiosity, continue and explored-grid losses on states s [..., S]; pads are
        ignored."""
        c, m = self.cfg, tgt["mask"]
        sparse_loss = self.two_hot.loss(self.sparse_reward_head(s), tgt["sparse"], m)
        standard_loss = self.two_hot.loss(self.standard_reward_head(s), tgt["standard"], m)
        cur = tgt["curiosity"]
        cur_w = torch.where(cur > 0, torch.full_like(cur, self.CURIOSITY_POS_WEIGHT), torch.ones_like(cur)) * m
        curiosity_loss = self.two_hot.loss(self.curiosity_head(s if c.curiosity_grad else s.detach()), cur, cur_w)
        # termination: rare cont = 0 frames are up-weighted
        cont_logit = self.continue_head(s).float()
        cont = tgt["cont"]
        cw = (1.0 + (c.continue_zero_weight - 1.0) * (1.0 - cont)) * m
        continue_loss = (F.binary_cross_entropy_with_logits(cont_logit, cont, reduction="none") * cw).sum() \
            / cw.sum().clamp(min=1.0)
        # explored 5x5 grid around Mario (needs the episode's history, so it reads [h, z]); unexplored cells
        # up-weighted. Its centre cell = "Mario's current tile was already visited" gates curiosity in dreams.
        grid = tgt["grid"].float()
        grid_logit = self.grid_head(s).float()
        gw = (1.0 + (c.grid_zero_weight - 1.0) * (1.0 - grid)) * m[..., None]
        grid_loss = (F.binary_cross_entropy_with_logits(grid_logit, grid, reduction="none") * gw).sum() \
            / gw.sum().clamp(min=1.0)
        with torch.no_grad():
            p_alive = self.alive_prob(None, logit=cont_logit)               # calibrated read, as in the dreams
            dead = (cont < 0.5).float() * m
            alive = (cont >= 0.5).float() * m
            nan = torch.tensor(float("nan"), device=p_alive.device)        # "no deaths in this batch"
            at_death = (p_alive * dead).sum() / dead.sum().clamp(min=1.0)
            hit = ((grid_logit > 0).float() == grid).float()
            denom = m.sum().clamp(min=1.0)
            stats = {"p_alive_at_death": torch.where(dead.sum() > 0, at_death, nan),
                     "p_alive_alive": (p_alive * alive).sum() / alive.sum().clamp(min=1.0),
                     "n_deaths": dead.sum(),
                     "grid_acc": (hit.mean(-1) * m).sum() / denom,
                     "grid_center_acc": (hit[..., GRID_CENTER_INDEX] * m).sum() / denom}
        return {"sparse": sparse_loss, "standard": standard_loss, "curiosity": curiosity_loss, "cont": continue_loss,
                "grid": grid_loss}, stats

    def _rollout(self, kv, z, post, batch):
        """Open-loop rollout from a random split point, fed its own prior samples: KL to real posteriors + states."""
        c, dev = self.cfg, self.device
        B, T = z.shape[:2]
        K = c.rollout_steps
        rows = torch.arange(B, device=dev).repeat_interleave(c.rollout_splits)
        R = rows.shape[0]
        n_real = batch["valid"].sum(1).long()[rows]
        p_max = torch.clamp(n_real - 1 - K, min=1, max=T - 1 - K)
        p = 1 + torch.minimum((torch.rand(R, device=dev) * p_max).long(), p_max - 1)
        state = self.tssm.open_loop_state(kv, rows, p)                     # real prefix K/V, with gradient
        zin = z[rows, p].detach()
        kls, states, match = [], [], []
        for j in range(K):
            t = p + j
            h = self.tssm.step_append(zin, batch["actions"][rows, t], state).float()
            raw = self.prior(h)
            target = post[rows, t + 1].detach()
            kls.append(kl_categorical(target, unimix_logprobs(raw, self.prior.unimix)))
            with torch.no_grad():
                zs = sample_onehot(unimix_logprobs(raw, c.dream_unimix), straight_through=False)
                match.append((zs.argmax(-1) == target.argmax(-1)).float().mean(-1))
                zin = zs.flatten(1)
            states.append(self.full_state(h, zin))
        t_idx = p[:, None] + 1 + torch.arange(K, device=dev)[None]          # frame predicted at step j
        M = batch["valid"][rows[:, None], t_idx]
        w = c.rollout_decay ** torch.arange(K, device=dev, dtype=torch.float32)
        kl = torch.stack(kls, dim=1)
        loss = (kl.clamp(min=c.free_nats) * M * w).sum() / (M * w).sum().clamp(min=1e-6)
        return {"loss": loss, "states": torch.stack(states, dim=1), "rows": rows, "t": t_idx, "mask": M,
                "match": torch.stack(match, dim=1), "kl": kl.detach()}

    # --- World model: update ----------------------------------------------------------------------------------------
    def train_world_model(self, batch, compute_metrics=True):
        """One end-to-end world-model update -> (states [B, T-1, S], kv context, start mask, metrics, start info)."""
        c = self.cfg
        B, T = batch["screen"].shape[:2]
        valid = batch["valid"]
        m1 = valid[:, 1:]
        tm = self.timer
        tm.start(compute_metrics)
        zero = torch.zeros((), device=self.device)
        self.wm_optimizer.zero_grad(set_to_none=True)
        measure = (c.grad_share_interval > 0 and self._grad_share_ok
                   and self.total_num_updates % c.grad_share_interval == 0)    # gradient-share log this update
        eager = eager_modules() if measure else contextlib.nullcontext()   # a graph that can be backpropagated per term
        with eager, self._amp():
            # 1. clean posteriors of every frame, straight-through one-hot samples
            flat = self._flat_obs(batch)
            if c.skip_pad_frames and "real_idx" in batch:                 # pads repeat their row's last real frame:
                ri = batch["real_idx"]                                     # its posterior is copied, not recomputed
                post = self.tokenizer.posterior({k: flat[k][ri] for k in self.OBS_KEYS})[batch["real_gather"]]
            else:
                post = self.tokenizer.posterior(flat)
            post = post.view(B, T, self.n_latents, self.classes)
            z = sample_onehot(post).flatten(-2)
            tm.mark("encode")

            # 2. dynamics: TSSM (z_t, a_t) -> h_{t+1} -> prior of z_{t+1}; balanced KL with free nats
            h, kv = self.tssm.forward_sequence(z[:, :-1], batch["actions"][:, :-1], return_kv=True, detach_kv=False)
            h = h.float()
            prior_raw = self.prior(h)
            prior = unimix_logprobs(prior_raw, self.prior.unimix)
            post1 = post[:, 1:]
            dyn = kl_categorical(post1.detach(), prior)
            rep = kl_categorical(post1, prior.detach())
            kl_dyn_term = c.w_dyn * masked_mean(dyn.clamp(min=c.free_nats), m1)    # trains prior + TSSM
            kl_rep_term = c.w_rep * masked_mean(rep.clamp(min=c.free_nats), m1)    # trains the encoder
            kl_loss = kl_dyn_term + kl_rep_term
            tm.mark("TSSM + prior fwd")

            # 3. heads on real states [h_{t+1}, z_{t+1}] and on dream-like states (open-loop rollout or prior samples)
            s_real = self.full_state(h, z[:, 1:])
            rows = torch.arange(B, device=self.device)[:, None]
            t_real = torch.arange(1, T, device=self.device)[None]
            tgt_real = self._targets(batch, rows, t_real)
            head_r, head_stats = self._head_losses(s_real, tgt_real)
            ro = None
            if c.rollout_steps > 0:
                ro = self._rollout(kv, z, post, batch)
                dream = {"states": ro["states"], "tgt": self._targets(batch, ro["rows"][:, None], ro["t"])}
                rollout_loss = ro["loss"]
            else:                                                          # one-step prior samples instead
                with torch.no_grad():
                    z1 = sample_onehot(unimix_logprobs(prior_raw, c.dream_unimix), straight_through=False)
                dream = {"states": self.full_state(h, z1.flatten(-2)), "tgt": tgt_real}
                rollout_loss = zero
            head_o, _ = self._head_losses(dream["states"], dream["tgt"])
            head = {k: (1.0 - c.w_dreamlike) * head_r[k] + c.w_dreamlike * head_o[k] for k in head_r}
            tm.mark("rollout + heads fwd")

            # 4. reconstruction (decoder reads z) + grounding (heads read z) on a subset of frames
            rec_stats, rec_parts = self._recon_branch(batch, z, dream)
            tm.mark("decode")

        # every world-model loss term times its fixed weight: their sum is the world-model loss
        terms = {"pixel": rec_parts["pixel"], "level": rec_parts["level"], "status": rec_parts["status"],
                 "kl_dyn": kl_dyn_term, "kl_rep": kl_rep_term, "rollout": c.w_rollout * rollout_loss,
                 "sparse_reward": c.w_sparse_reward * head["sparse"],
                 "standard_reward": c.w_standard_reward * head["standard"],
                 "continue": c.w_continue * head["cont"], "grid": c.w_grid * head["grid"],
                 "curiosity": c.w_curiosity * head["curiosity"]}
        world_model_loss = sum(terms.values())
        self._accumulate_wm_losses(terms)
        if measure:
            with eager_modules():
                self._measure_grad_shares(terms)                           # diagnostic: one backward per term
        # loss groups of the (optional) harmony weights; the curiosity head (it trains only itself) keeps its weight
        groups = {"pixel": terms["pixel"], "grounding": terms["level"] + terms["status"],
                  "kl": terms["kl_dyn"] + terms["kl_rep"], "rollout": terms["rollout"],
                  "reward": terms["sparse_reward"] + terms["standard_reward"], "cont": terms["continue"],
                  "grid": terms["grid"]}
        fixed = terms["curiosity"]
        hw = None
        if c.harmony:
            hw = self.harmony.weights()
            wm_opt_loss = sum(hw[k] * v for k, v in groups.items()) + fixed
        else:
            wm_opt_loss = world_model_loss
        clip = self.wm_clipper.threshold()
        grad_norm = self._optimize(wm_opt_loss, "wm_optimizer", self.wm_params, clip)
        if not c.sync_free:
            grad_norm = float(grad_norm)
        self.wm_clipper.record(grad_norm, clip)
        if c.harmony:
            self.harmony_optimizer.zero_grad(set_to_none=True)
            self.harmony.scale_loss({k: v.detach() for k, v in groups.items()}).backward()
            self.harmony_optimizer.step()
        tm.mark("WM backward + step")
        self._update_drift()

        start_mask = (m1 * (batch["continues"][:, 1:] > 0).float()).detach()
        with torch.no_grad():                                              # distance to the next death + surprise
            cont, idx = batch["continues"], torch.arange(T, device=self.device)[None].expand(B, T)
            ends = batch.get("episode_end")                                # last row of its episode (no flagpoles)
            if ends is None:                                               # hand-made batches: last valid row
                ends = torch.cat([valid[:, 1:], torch.zeros_like(valid[:, :1])], dim=1) <= 0
            death = (cont <= 0) & (valid > 0) & ends                       # terminal that ends its episode
            pos = torch.where(death, idx, torch.full_like(idx, 4 * T))
            next_death = torch.flip(torch.cummin(torch.flip(pos, [1]), dim=1).values, [1])
            start_info = {"death_dist": (next_death - idx)[:, 1:], "surprise": dyn.detach().float() * m1}
            if c.loss_sample_prob > 0 and "index" in batch:                # masked on the CPU: fixed-size GPU tensors
                if c.sync_free and "index_cpu" in batch and self.device.type == "cuda":
                    self._defer_priorities(batch, dyn)
                else:
                    sel = (m1 > 0).cpu()
                    self.buffer.update_priorities(batch["index"][:, 1:].cpu()[sel], dyn.detach().float().cpu()[sel])
        if compute_metrics or self.total_num_updates % self.DEATH_PROBE_EVERY == 0:
            self._accumulate_death_probes(h, prior_raw, z, kv, batch, death)

        metrics = {}
        if compute_metrics:
            with torch.no_grad():
                metrics = {"world_model_loss_last": world_model_loss.item(), "kl_dyn": masked_mean(dyn, m1).item(),
                           "kl_loss": kl_loss.item(), "rollout_loss": rollout_loss.item(),
                           "wm_grad_norm": float(grad_norm), "padded_fraction": 1.0 - valid.mean().item(),
                           "encoder_drift": self.last_drift}
                metrics.update({k: float(v) for k, v in rec_stats.items()})
                metrics.update({f"{k}_loss" if k != "cont" else "continue_loss": v.item() for k, v in head.items()})
                metrics["reward_loss"] = metrics["sparse_loss"] + metrics["standard_loss"]
                metrics.update({k: v.item() for k, v in head_stats.items()})
                mode_post = post.argmax(-1)
                metrics["post_entropy"] = masked_mean(entropy_categorical(post), valid).item()
                metrics["prior_entropy"] = masked_mean(entropy_categorical(prior), m1).item()
                metrics["latent_usage"] = float(class_usage(z.detach(), self.n_latents, self.classes, valid))
                metrics["pred_match"] = masked_mean((prior.argmax(-1) == mode_post[:, 1:]).float().mean(-1), m1).item()
                metrics["copy_match"] = masked_mean((mode_post[:, :-1] == mode_post[:, 1:]).float().mean(-1), m1).item()
                if ro is not None:
                    k = ro["match"].shape[1] - 1
                    metrics["rollout_match_k"] = masked_mean(ro["match"][:, k], ro["mask"][:, k]).item()
                    metrics["rollout_kl_k"] = masked_mean(ro["kl"][:, k], ro["mask"][:, k]).item()
                clip_threshold, clip_fraction = self.wm_clipper.stats()
                metrics.update({"wm_clip_threshold": clip_threshold, "wm_clip_fraction": clip_fraction})
                metrics.update({f"group_{k}": float(v.detach()) for k, v in groups.items()})
                if hw is not None:
                    metrics.update({f"harmony_{k}": float(v) for k, v in hw.items()})
                    metrics["harmonized_loss"] = float(wm_opt_loss.detach())
                lo, hi = c.death_dream_window
                dd = start_info["death_dist"]
                metrics["pre_death_states"] = float(((dd >= lo) & (dd <= hi)).float().mul(start_mask).sum())
                metrics.update(self._death_probe_means())
                metrics.update(self._wm_loss_means())                     # world_model_loss + wm_* (round means)
                metrics.update(self.last_grad_shares)                     # newest gradient-share measurement
            tm.mark("metrics")
        kv_context = [(k.detach(), v.detach()) for k, v in kv]
        return s_real.detach(), kv_context, start_mask, metrics, start_info

    GRAD_SHARE_GROUPS = ("tssm", "encoder")

    def _measure_grad_shares(self, terms):
        """Diagnostic (no training effect): the gradient norm of every weighted loss term, computed separately
        (torch.autograd.grad, the graph is kept) on the TSSM parameters and on the encoder parameters, and each term's
        share of the sum of the norms of its group -> self.last_grad_shares ("gnorm_<group>_<term>",
        "gshare_<group>_<term>"). A failure disables the measurement with one message; training goes on."""
        groups = {"tssm": [p for p in self.tssm.parameters() if p.requires_grad],
                  "encoder": [p for p in self.tokenizer.encoder.parameters() if p.requires_grad]}
        params = groups["tssm"] + groups["encoder"]
        n_t = len(groups["tssm"])
        zero = torch.zeros((), device=self.device)
        try:
            norms = {}
            for name in self.WM_LOSS_TERMS:
                loss = terms[name]
                if not (torch.is_tensor(loss) and loss.requires_grad):
                    norms[name] = (zero, zero)
                    continue
                grads = torch.autograd.grad(loss.float(), params, retain_graph=True, allow_unused=True)

                def norm(gs):
                    sq = [g.float().square().sum() for g in gs if g is not None]
                    return torch.stack(sq).sum().sqrt() if sq else zero
                norms[name] = (norm(grads[:n_t]), norm(grads[n_t:]))
                del grads
            vals = torch.stack([torch.stack(v) for v in norms.values()]).tolist()        # one host sync
        except Exception as e:                                             # never let a diagnostic stop training
            print(f"[!] gradient-share log disabled ({type(e).__name__}: {e})")
            self._grad_share_ok = False
            return
        out = {"grad_share_update": float(self.total_num_updates)}
        for gi, group in enumerate(self.GRAD_SHARE_GROUPS):
            total = sum(v[gi] for v in vals)
            for name, v in zip(norms, vals):
                out[f"gnorm_{group}_{name}"] = v[gi]
                out[f"gshare_{group}_{name}"] = v[gi] / total if total > 0 else 0.0
        self.last_grad_shares = out

    @torch.no_grad()
    def _accumulate_wm_losses(self, terms):
        """Adds the weighted loss terms of one update to the round sums (on the device: no host sync)."""
        vals = torch.stack([terms[k].detach().float() for k in self.WM_LOSS_TERMS])
        if self._wm_loss_acc is None:
            self._wm_loss_acc = torch.zeros(len(self.WM_LOSS_TERMS) + 1, device=vals.device)
        self._wm_loss_acc[:-1] += vals
        self._wm_loss_acc[-1] += 1.0

    def _wm_loss_means(self):
        """Round averages of the weighted loss terms ("wm_<term>") and of their sum ("world_model_loss"); resets."""
        if self._wm_loss_acc is None or float(self._wm_loss_acc[-1]) == 0.0:
            return {}
        acc = self._wm_loss_acc.tolist()
        n = acc[-1]
        out = {f"wm_{k}": v / n for k, v in zip(self.WM_LOSS_TERMS, acc[:-1])}
        out["world_model_loss"] = sum(acc[:-1]) / n
        out["wm_updates"] = n
        self._wm_loss_acc.zero_()
        return out

    # --- Open-loop probes (diagnostics) -----------------------------------------------------------------------------
    @torch.no_grad()
    def _accumulate_death_probes(self, h, prior_raw, z, kv, batch, death):
        """Adds p(alive) at the real deaths of a batch, read on dream-side states, to the round sums.

        pred1: [h_t, z^_t], z^_t sampled from the prior given the real history (one dream step).
        dream: DEATH_PROBE_STEPS open-loop steps on the model's own samples with the real actions, ending at the death
        frame; survival: product of p(alive) over those steps. Logged as pAliveDeathPred / pAliveDeathDream /
        dreamDeathSurvival. A model whose dreams die where the real game did gives values near 0."""
        c, dev = self.cfg, self.device
        if self._death_probe_acc is None:
            self._death_probe_acc = torch.zeros(5, device=dev)            # pred1 sum, n, dream sum, survival sum, n
        acc = self._death_probe_acc
        rows, t = torch.nonzero(death[:, 1:], as_tuple=True)
        if rows.numel() == 0:
            return
        t = t + 1                                                          # observation index of each death
        alive = self.alive_prob                                            # calibrated, as in the dreams
        sample = lambda raw: sample_onehot(unimix_logprobs(raw, c.dream_unimix),  # noqa: E731
                                           straight_through=False).flatten(-2)
        with self._amp():
            z1 = sample(prior_raw[rows, t - 1].detach().float())
            pa1 = alive(self.full_state(h[rows, t - 1].detach(), z1))
            acc[0] += pa1.sum()
            acc[1] += pa1.numel()
            k = self.DEATH_PROBE_STEPS
            ok = t - k >= 1                                                # a real prefix before the open loop
            if bool(ok.any()):
                r, p = rows[ok], t[ok] - k
                state = self.tssm.open_loop_state([(a.detach(), b.detach()) for a, b in kv], r, p)
                zz = z[r, p].detach()
                surv = torch.ones(r.shape[0], device=dev)
                for j in range(k):                                         # step j predicts frame p + j + 1
                    hj = self.tssm.step_append(zz, batch["actions"][r, p + j], state).float()
                    zz = sample(self.prior(hj))
                    pa = alive(self.full_state(hj, zz))
                    surv = surv * pa
                acc[2] += pa.sum()                                         # at the death frame
                acc[3] += surv.sum()
                acc[4] += pa.numel()

    def _death_probe_means(self):
        """Round averages of the death probe (NaN without deaths); resets the sums."""
        nan = float("nan")
        out = {"p_alive_death_pred1": nan, "p_alive_death_dream": nan, "dream_death_survival": nan,
               "death_probe_deaths": 0.0}
        if self._death_probe_acc is not None:
            s = self._death_probe_acc.tolist()
            if s[1] > 0:
                out["p_alive_death_pred1"] = s[0] / s[1]
            if s[4] > 0:
                out["p_alive_death_dream"] = s[2] / s[4]
                out["dream_death_survival"] = s[3] / s[4]
            out["death_probe_deaths"] = s[1]
            self._death_probe_acc.zero_()
        return out

    @torch.no_grad()
    def context_kv(self, zf, actions):
        """Teacher-forced pass over real latents -> (h [B, T-1, d], kv)."""
        with self._amp():
            h, kv = self.tssm.forward_sequence(zf[:, :-1], actions[:, :-1], return_kv=True)
        return h.float(), kv

    @torch.no_grad()
    def dream_step(self, z, action, cache):
        """One eager imagined step -> (h, one-hot sample z, prior log-probs)."""
        h = self.tssm.forward_step(z, action, cache).float()
        logp = unimix_logprobs(self.prior(h), self.cfg.dream_unimix)
        return h, sample_onehot(logp, straight_through=False).flatten(1), logp

    @torch.no_grad()
    def open_loop(self, zf, actions, kv, valid, n_rows=16, horizon=15, t0=None, rows=None):
        """Open-loop generations from real contexts: recorded actions (two seeds), RIGHT held, LEFT held."""
        B, T = valid.shape
        lens = valid.sum(1).long()
        if rows is None:
            cand = torch.nonzero(lens >= horizon + 2).flatten()
            if cand.numel() == 0:
                return None
            rows = cand[torch.randperm(cand.numel(), device=cand.device)[:n_rows]]
        n = rows.shape[0]
        if t0 is None:
            span = (lens[rows] - 1 - horizon).clamp(min=1)
            t0 = 1 + (torch.rand(n, device=self.device) * span).long().clamp(max=span - 1)
        steps = t0[:, None] + torch.arange(horizon + 1, device=self.device)[None]
        real = zf[rows[:, None], steps]
        runs = {}
        fixed = {"right": ACTION_RIGHT, "left": ACTION_LEFT}
        with self._amp():
            for name in ("teacher", "teacher2", "right", "left"):
                cache = self.tssm.make_cache(n, self.device, self.cache_dtype)
                self.tssm.prefill(cache, kv, rows, t0 - 1)
                z = zf[rows, t0]
                out = []
                for k in range(horizon):
                    if name in fixed:
                        a = F.one_hot(torch.full((n,), fixed[name], device=self.device), self.action_dim).float()
                    else:
                        a = actions[rows, t0 + k]
                    _, z, _ = self.dream_step(z, a, cache)
                    out.append(z.float())
                runs[name] = torch.stack(out, dim=1)
        return {"rows": rows, "t0": t0, "real": real, **runs}

    # --- Imagination ------------------------------------------------------------------------------------------------
    def _dream_graph_for(self, n):
        """CUDA graph of one imagined step for n dreams, or None to run eagerly (CPU, disabled or capture failed)."""
        if not self._use_dream_graph or self.device.type != "cuda":
            return None
        g = self._dream_graph
        if g is not None and g.n == n:
            return g
        self._dream_graph = None
        try:
            g = GraphedDreamStep(self.tssm, self.prior, n, self.cache_dtype, self.amp_dtype, self.device)
        except Exception as e:                                             # never let the graph stop training
            print(f"[!] dream CUDA graph disabled ({type(e).__name__}: {e}); dreams run eagerly.")
            self._use_dream_graph = False
            return None
        self._dream_graph = g
        return g

    def _dream_starts(self, start_mask, start_info, B, T, dev):
        """Start (row, t0) of every dream: uniform, pre-death (each dream with probability death_dream_prob),
        surprising, or first-action branch groups."""
        c, K, A = self.cfg, self.cfg.dreams_per_sequence, self.action_dim
        N = B * K
        rows = torch.arange(B, device=dev).repeat_interleave(K)
        elig = start_mask.float() if start_mask is not None else torch.ones(B, T, device=dev)
        e = elig[rows]
        counts = e.sum(1, keepdim=True)
        t0 = torch.multinomial(torch.where(counts > 0, e, torch.ones_like(e)), 1).squeeze(1)
        forced = torch.full((N,), -1, dtype=torch.long, device=dev)       # forced first action (-1 = policy)
        pre = torch.zeros(B, T, device=dev)
        G, n_norm = 0, N
        if start_info is not None:
            lo, hi = c.death_dream_window
            pre = ((start_info["death_dist"] >= lo) & (start_info["death_dist"] <= hi)).float() * elig
            floor = 1e-6 * elig.flatten() + 1e-12                          # empty pool -> uniform over eligible
            G = min(int(c.branch_groups), (N // 2) // A)
            n_norm = N - G * A
            if c.death_dream_prob > 0:                                     # coin flip per dream (no host sync)
                death_m = torch.rand(n_norm, device=dev) < c.death_dream_prob
                idx = torch.multinomial(pre.flatten() + floor, n_norm, replacement=True)
                rows[:n_norm] = torch.where(death_m, idx // T, rows[:n_norm])
                t0[:n_norm] = torch.where(death_m, idx % T, t0[:n_norm])
            else:
                death_m = torch.zeros(n_norm, dtype=torch.bool, device=dev)
            n_l = min(int(round(c.dream_loss_frac * n_norm)), n_norm)
            if n_l > 0:                                                    # the first n_l dreams not pre-death
                sur = start_info["surprise"].float().clamp_min(0.0) * elig
                idx = torch.multinomial(sur.flatten() + floor, n_l, replacement=True)
                rank = torch.cumsum((~death_m).long(), 0) - 1
                loss_m = ~death_m & (rank < n_l)
                pick = idx[rank.clamp(0, n_l - 1)]
                rows[:n_norm] = torch.where(loss_m, pick // T, rows[:n_norm])
                t0[:n_norm] = torch.where(loss_m, pick % T, t0[:n_norm])
            if G > 0:                                                      # each group tries every first action
                idx = torch.multinomial(pre.flatten() + floor, G, replacement=False)
                rows[n_norm:] = (idx // T).repeat_interleave(A)
                t0[n_norm:] = (idx % T).repeat_interleave(A)
                forced[n_norm:] = torch.arange(A, device=dev).repeat(G)
        start_ok = (elig[rows, t0] > 0).float()
        return rows, t0, start_ok, forced, n_norm, G, pre

    @torch.no_grad()
    def _combine_advantages(self, adv, cur_adv, lam, cur_lam, weights):
        """Combined actor advantage of the extrinsic and curiosity streams [N, H] -> (combined, info).

        "std" (the only mode): each stream centred on its own survival-weighted mean and divided by its own
        survival-weighted std (at least ADV_STD_MIN), then ext + curiosity_scale x cur. The sum is not normalised again,
        so curiosity_scale is the curiosity stream's weight relative to the extrinsic one, whatever their reward sizes.
        info: the two per-stream stds (the divisors, before the floor), the sum's std, and per stream its
        contribution's std, this batch's return range (P95 - P5) and its share of the sum's variance,
        cov(A_k, A) / var(A) (the shares add to 1; negative = the stream works against the total)."""
        c = self.cfg
        m_e, s_ext = weighted_moments(adv, weights)
        m_c, s_cur = weighted_moments(cur_adv, weights)
        parts = {"ext": (adv.float() - m_e) / s_ext.clamp(min=ADV_STD_MIN),
                 "cur": c.curiosity_scale * (cur_adv.float() - m_c) / s_cur.clamp(min=ADV_STD_MIN)}
        combined = parts["ext"] + parts["cur"]
        mean, std = weighted_moments(combined, weights)
        wsum = weights.sum().clamp(min=1e-8)
        info = {"scale_ext": s_ext, "scale_cur": s_cur, "combined_std": std}
        q = torch.tensor([0.05, 0.95], device=adv.device)
        for name, a, ret in (("ext", parts["ext"], lam), ("cur", parts["cur"], cur_lam)):
            m, sigma = weighted_moments(a, weights)
            r = torch.quantile(ret.detach().flatten().float(), q)
            cov = (weights * (a - m) * (combined - mean)).sum() / wsum
            info[name] = {"adv_std": sigma, "return_range": r[1] - r[0],
                          "adv_share": cov / std.square().clamp(min=1e-12)}
        return combined, info

    def imagine(self, full_state, kv_context, start_mask=None, start_info=None, real=None, horizon=None,
                compute_metrics=True, pack_dreams=True):
        """Actor-critic update on dreams started from real states -> (metrics, best dream, random dream)."""
        c = self.cfg
        horizon = horizon or c.dream_horizon
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.curiosity_critic_optimizer.zero_grad(set_to_none=True)
        tm = self.timer
        tm.start(compute_metrics, reset=False)

        all_states = full_state.detach()
        B, T = all_states.shape[:2]
        dev = all_states.device
        A = self.action_dim
        rows, t0, start_ok, forced, n_norm, G, pre = self._dream_starts(start_mask, start_info, B, T, dev)
        num_dreams = rows.shape[0]
        start_states = all_states[rows, t0]
        graph = self._dream_graph_for(num_dreams)
        cache = graph.cache if graph is not None else self.tssm.make_cache(num_dreams, dev, self.cache_dtype)
        self.tssm.prefill(cache, kv_context, rows, t0)                     # real context in the cache

        with self._amp():
            # --- roll out the policy in the world model
            states = [start_states]
            log_probs, entropies, acts = [], [], []
            curr = start_states
            z = start_states[:, self.tssm_dim:]
            probs0 = None
            for i in range(horizon):
                action, logprob, entropy, probs = self.actor(curr, return_probs=True)
                if i == 0:
                    probs0 = probs.detach()
                    if G > 0:                                              # branch dreams: given first action
                        a_f = F.one_hot(forced.clamp(min=0), A).float()
                        is_b = forced >= 0
                        action = torch.where(is_b[:, None], a_f, action)
                        logprob = torch.where(is_b, (probs.clamp_min(1e-8).log() * a_f).sum(-1), logprob)
                with torch.no_grad():
                    if graph is not None:
                        h, z, _ = graph(action.detach(), z, unimix=c.dream_unimix)
                    else:
                        h, z, _ = self.dream_step(z, action.detach(), cache)
                    curr = torch.nan_to_num(self.full_state(h, z), nan=0.0, posinf=1e4, neginf=-1e4)
                states.append(curr)
                log_probs.append(logprob)
                entropies.append(entropy)
                acts.append(action)
            states = torch.stack(states, dim=1)
            log_probs = torch.stack(log_probs, dim=1)
            entropies = torch.stack(entropies, dim=1)
            acts = torch.stack(acts, dim=1)

            # --- predicted rewards, curiosity and survival
            with torch.no_grad():
                nxt = states[:, 1:]
                pred_rewards = (self.two_hot.decode(self.sparse_reward_head(nxt))
                                + self.two_hot.decode(self.standard_reward_head(nxt))).squeeze(-1)
                pred_cur = self.two_hot.decode(self.curiosity_head(nxt)).squeeze(-1)
                gate = 1.0 - torch.sigmoid(self.grid_head(nxt).float()[..., GRID_CENTER_INDEX])  # visited tile: 0
                pred_cur = pred_cur * gate
                pred_alive = self.alive_prob(nxt)                          # calibrated p(alive) of s_k+1
                weights = torch.cat([torch.ones_like(pred_alive[:, :1]),
                                     torch.cumprod(pred_alive, dim=1)[:, :-1]], dim=1) * start_ok[:, None]
                cont = c.continue_discount * pred_alive
                # death penalty where the dream's survival drops
                r_death = c.death_penalty * (1.0 - pred_alive) if c.death_penalty else torch.zeros_like(pred_alive)
                pred_rewards = pred_rewards + r_death

            # --- returns and advantages
            imagined = states.detach()
            critic_logits = self.critic(imagined)
            cur_critic_logits = self.curiosity_critic(imagined)
            with torch.no_grad():
                values = self.two_hot.decode(self.critic_target(imagined)).squeeze(-1)
                cur_values = self.two_hot.decode(self.curiosity_critic_target(imagined)).squeeze(-1)
                lam = lambda_returns(pred_rewards, values, cont)
                cur_lam = lambda_returns(pred_cur, cur_values, cont)
                adv = lam - values[:, :-1]
                cur_adv = cur_lam - cur_values[:, :-1]
                crit_w = weights
                if G > 0:                                                  # branch step 0: group-mean baseline
                    for a_, l_ in ((adv, lam), (cur_adv, cur_lam)):
                        g_ = l_[n_norm:, 0].view(G, A)
                        a_[n_norm:, 0] = (g_ - g_.mean(1, keepdim=True)).flatten()
                    crit_w = weights.clone()
                    crit_w[n_norm:, 0] = 0.0
                combined, adv_info = self._combine_advantages(adv, cur_adv, lam, cur_lam, weights)
                s_ext, s_cur = adv_info["scale_ext"], adv_info["scale_cur"]
                forced_first = None
                if G > 0:                                                  # branch dreams: their forced first action
                    forced_first = (n_norm, probs0[n_norm:].gather(1, forced[n_norm:, None]).squeeze(1))

            # --- losses (REINFORCE actor)
            actor_loss, _ = reinforce_loss(c, combined, log_probs, entropies, weights, forced_first)
            critic_loss = self.two_hot.loss(critic_logits[:, :-1], lam, crit_w)
            cur_critic_loss = self.two_hot.loss(cur_critic_logits[:, :-1], cur_lam, crit_w)

            # replay value loss: the critic also regresses lambda-returns of real rewards along the real sequence
            use_repval = real is not None and c.w_repval > 0 and T > 1
            if use_repval:
                rc, rv = real["continues"].float(), real["valid"].float()
                with torch.no_grad():
                    real_values = self.two_hot.decode(self.critic_target(all_states)).squeeze(-1)
                    real_rew = real["rewards"][:, :T - 1].float()
                    step_ok = rv[:, 1:] * (rc[:, :T - 1] > 0).float()      # next frame real, source alive
                    real_lam = lambda_returns(real_rew, real_values, c.continue_discount * rc[:, 1:], valid=step_ok)
                real_logits = self.critic(all_states[:, :-1])
                repval_loss = self.two_hot.loss(real_logits, real_lam, step_ok)
            else:
                repval_loss = torch.zeros((), device=dev)

        self._optimize(critic_loss + c.w_repval * repval_loss, "critic_optimizer", self.critic.parameters(), 1.0)
        self._optimize(cur_critic_loss, "curiosity_critic_optimizer", self.curiosity_critic.parameters(), 1.0)
        self._optimize(actor_loss, "actor_optimizer", self.actor.parameters(), 1.0)
        self._critic_ema_update()
        tm.mark("dream + actor/critic")

        metrics = {}
        if compute_metrics:
            metrics = {
                "actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(),
                "curiosity_critic_loss": cur_critic_loss.item(), "entropies": entropies.mean().item(),
                "advantages": adv.mean().item(), "curiosity_advantages": cur_adv.mean().item(),
                "critic_values": self.two_hot.decode(critic_logits.detach()).mean().item(),
                "target_critic_values": values.mean().item(),
                "curiosity_critic_values": self.two_hot.decode(cur_critic_logits.detach()).mean().item(),
                "advantage_pos_fraction": (combined >= 0).float().mean().item(),
                "curiosity_flip_fraction": ((combined >= 0) != (adv >= 0)).float().mean().item(),
                "dream_mean_curiosity": pred_cur.mean().item(),
                "dream_mean_alive": pred_alive.mean().item(),
                "dream_death_fraction": (weights[:, -1] < 0.5).float().mean().item(),
                "replay_value_loss": repval_loss.item(),
                "replay_critic_values": self.two_hot.decode(real_logits.detach()).mean().item() if use_repval else 0.0,
                "replay_returns": masked_mean(real_lam, step_ok).item() if use_repval else 0.0,
                "adv_scale": float(s_ext),                                 # the reward stream's advantage std
                "cur_adv_scale": float(s_cur),                             # the curiosity stream's advantage std
                **actor_metrics(combined, weights),
                "dream_start_pre_death": pre[rows, t0].mean().item(),
                "adv_combined_std": float(adv_info["combined_std"]),      # the combined advantage's std
                "dream_death_penalty": masked_mean(r_death, weights).item(),   # per dream step, survival-weighted
            }
            with torch.no_grad():                                          # per stream: is each critic learning?
                for name, logits, target in (("ext", critic_logits, lam), ("cur", cur_critic_logits, cur_lam)):
                    pred = self.two_hot.decode(logits[:, :-1].detach()).squeeze(-1)
                    metrics[f"{name}_critic_ev"] = explained_variance(pred, target, crit_w)
                    for k in ("adv_std", "return_range", "adv_share"):
                        metrics[f"{name}_{k}"] = float(adv_info[name][k])
            if G > 0:
                c0 = combined[n_norm:, 0].view(G, A)
                pol = probs0[n_norm::A].argmax(-1)                         # policy's choice at each group start
                metrics["branch_agree"] = (c0.argmax(1) == pol).float().mean().item()
                metrics["branch_spread"] = (c0.max(1).values - c0.min(1).values).mean().item()
        if not pack_dreams:
            return metrics, None, None

        def pack(i, metric, label):
            """Dream i as CPU tensors for the PDF."""
            return (metric, states[i].detach().float().cpu(), pred_rewards[i].detach().cpu(), lam[i].detach().cpu(),
                    acts[i].detach().cpu(), adv[i].detach().cpu(), cur_adv[i].detach().cpu(),
                    combined[i].detach().cpu(), label)

        traj = (combined * weights).sum(dim=1)
        if c.sync_free:                                                    # GPU copies; to CPU when the round ends
            def keep(i, metric, label):
                return (metric,) + tuple(x[i].detach().clone() for x in (states, pred_rewards, lam, acts, adv,
                                                                         cur_adv, combined)) + (label,)
            best_idx = torch.argmax(traj)
            rand_idx = torch.randint(0, traj.shape[0], (1,)).item()        # CPU generator: no sync
            return metrics, keep(best_idx, traj[best_idx].detach().clone(), "MaxAdvantage"), \
                keep(rand_idx, traj[rand_idx].detach().clone(), "Random")
        best_idx = torch.argmax(traj).item()
        rand_idx = torch.randint(0, traj.shape[0], (1,)).item()
        return metrics, pack(best_idx, traj[best_idx].item(), "MaxAdvantage"), \
            pack(rand_idx, traj[rand_idx].item(), "Random")

    # --- Dream audit (diagnostics) ----------------------------------------------------------------------------------
    def audit(self, seed=None):
        """Dream audit: from the last play's snapshots, the same states are played audit_horizon steps in the dream and
        in the real game with the actor's greedy action -> metrics ({} without a usable snapshot). Uses its own seeded
        generators and its own emulator; trains nothing."""
        c, t_start = self.cfg, time.perf_counter()
        if c.audit_horizon <= 0:
            return {}
        if seed is None:
            seed = int(np.random.SeedSequence([c.seed, 0xA4D17, self.total_num_updates]).generate_state(1)[0])
        items = self.audit_snapshots.usable(self.buffer, c.context_length)
        picks = select_starts(items, c.audit_starts, np.random.default_rng(seed))
        if not picks:
            return {}
        starts = [items[i] for i, _ in picks]
        kinds = [AUDIT_KINDS.index(k) for _, k in picks]
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        with torch.no_grad(), eager_modules():
            ctx = self.audit_context([s["row"] for s in starts], [s["step"] for s in starts], gen)
            dream = self.audit_dream(ctx, c.audit_horizon, gen)
            real = self.audit_real(ctx, [s["snapshot"] for s in starts], c.audit_horizon, gen)
            dream.update(self._audit_compare(dream, real, ctx))
        metrics = audit_metrics(dream, real, kinds, c.continue_discount)
        metrics["seconds"] = time.perf_counter() - t_start
        return metrics

    def _audit_sample(self, logp, gen):
        """One-hot samples [..., L*K] of log-probs [..., L, K] (Gumbel-max) with the audit's generator."""
        u = torch.rand(logp.shape, generator=gen, device=logp.device).clamp_(1e-6, 1.0 - 1e-6)
        return sample_onehot(logp.float(), straight_through=False, gumbel=-torch.log(-torch.log(u))).flatten(-2)

    def _audit_encode(self, obs, gen, chunk=1024):
        """Posterior samples [N, L*K] of encoder inputs {key: [N, ...]} with the audit's generator."""
        n = obs["screen"].shape[0]
        zs = []
        for i in range(0, n, chunk):
            with self._amp():
                logp = self.tokenizer.posterior({k: obs[k][i:i + chunk] for k in self.OBS_KEYS})
            zs.append(self._audit_sample(logp, gen))
        return torch.cat(zs)

    def _buffer_obs(self, rows):
        """Encoder inputs (+ sprite overlay) of replay rows [N] on the device."""
        b, dev = self.buffer, self.device
        return {"screen": b.screens[rows].to(dev), "palette": b.palettes[rows].to(dev),
                "grid": unpack_bits(b.grids[rows].to(dev), GRID_DIM), "map_id": b.map_ids[rows].long().to(dev),
                "ram": b.rams[rows].to(dev), "blocks": b.blocks[rows].long().to(dev)}

    def _audit_action(self, s):
        """The actor's greedy one-hot action for states s."""
        return self.actor(s, greedy=True)[0].float()

    @torch.no_grad()
    def audit_context(self, rows, steps, gen):
        """Real context of N starts (replay rows, episode steps) -> {"kv", "t0" [N], "start" [N, S], "obs0"}: the
        context_length frames before each start encoded and run through the TSSM, the start frame's latent, and the
        start frame itself (its reconstruction is the audit's error floor)."""
        c, buf, dev = self.cfg, self.buffer, self.device
        W = c.context_length
        rows = torch.as_tensor(rows, dtype=torch.long)
        N = rows.shape[0]
        n_ctx = torch.as_tensor(steps, dtype=torch.long).clamp(min=1, max=W)          # tokens before the start
        j = torch.arange(W + 1)[None]
        src = (rows[:, None] - n_ctx[:, None] + torch.minimum(j, n_ctx[:, None])) % buf.capacity   # pads: the start
        obs = self._buffer_obs(src.flatten())
        zf = self._audit_encode(obs, gen).view(N, W + 1, self.latent_dim)
        actions = F.one_hot(buf.actions[src].long(), self.action_dim).float().to(dev)
        h, kv = self.context_kv(zf, actions)                                           # h[:, j]: after token j
        idx = torch.arange(N, device=dev)
        t0 = (n_ctx - 1).to(dev)
        start = torch.nan_to_num(self.full_state(h[idx, t0], zf[idx, t0 + 1]), nan=0.0, posinf=1e4, neginf=-1e4)
        obs0 = self._buffer_obs(rows % buf.capacity)
        return {"kv": kv, "t0": t0, "start": start, "obs0": obs0}

    def _audit_heads(self, states):
        """Per-stream rewards [N, H, 4] (sparse, standard, gated curiosity, death penalty) and calibrated p(alive)
        [N, H] of the new states states[:, 1:], as imagine computes them."""
        c, nxt = self.cfg, states[:, 1:]
        sparse = self.two_hot.decode(self.sparse_reward_head(nxt)).squeeze(-1)
        standard = self.two_hot.decode(self.standard_reward_head(nxt)).squeeze(-1)
        gate = 1.0 - torch.sigmoid(self.grid_head(nxt).float()[..., GRID_CENTER_INDEX])
        cur = self.two_hot.decode(self.curiosity_head(nxt)).squeeze(-1) * gate
        alive = self.alive_prob(nxt)
        death = c.death_penalty * (1.0 - alive)
        return torch.stack([sparse, standard, cur, death], dim=-1).float(), alive

    @torch.no_grad()
    def audit_dream(self, ctx, horizon, gen):
        """The audit's dreams from audit_context with the greedy action -> {"r" [N, H, 4], "alive" [N, H], "frames"
        [N, H + 1, 3, 96, 120] (decoded, step 0 = the start latent), "level" / "tier" [N, H] read from the dreamed
        latents, "grid" [N, H, 25] from the grid head}."""
        c, dev, N = self.cfg, self.device, ctx["start"].shape[0]
        cache = self.tssm.make_cache(N, dev, self.cache_dtype)
        self.tssm.prefill(cache, ctx["kv"], torch.arange(N, device=dev), ctx["t0"])
        curr = ctx["start"]
        z = curr[:, self.tssm_dim:]
        states = [curr]
        with self._amp():
            for _ in range(horizon):
                h = self.tssm.forward_step(z, self._audit_action(curr), cache).float()
                z = self._audit_sample(unimix_logprobs(self.prior(h), c.dream_unimix), gen)
                curr = torch.nan_to_num(self.full_state(h, z), nan=0.0, posinf=1e4, neginf=-1e4)
                states.append(curr)
            states = torch.stack(states, dim=1)                                    # [N, H + 1, S]
            r, alive = self._audit_heads(states)
            zs = states[..., self.tssm_dim:]
            frames = torch.cat([torch.sigmoid(self.tokenizer.decode(zz).float())
                                for zz in zs.flatten(0, 1).split(512)]).view(N, horizon + 1, 3, IMG_H, IMG_W)
            level, tier = self.tokenizer.heads.predict(zs[:, 1:])
            grid = self.grid_head(states[:, 1:]).float() > 0
        return {"r": r, "alive": alive, "frames": frames, "level": level, "tier": tier, "grid": grid}

    def _audit_envs(self):
        """Emulators for the real rollouts: the training envs when they can restore snapshots (each start restored
        once, then played in lockstep like play(); play() resets them afterwards), else one own headless emulator
        that plays the starts one after another."""
        if self.envs is not None and hasattr(self.envs, "restore"):
            return self.envs
        if self._audit_env is None:
            from .env import SequentialMarioEnvs, SuperMarioEnv
            rom = getattr(self.envs, "rom_path", "game/SuperMarioBrosDeluxe.gbc")
            state = getattr(self.envs, "state_path", "game/SMBDeluxe.Start.state")
            self._audit_env = SequentialMarioEnvs([SuperMarioEnv(
                rom_path=rom, state_path=state, window="null", verbose=False, frames_per_step=self.cfg.frames_per_step,
                skip_transitions=getattr(self.envs, "skip_transitions", self.cfg.skip_transitions))])
        return self._audit_env

    @torch.no_grad()
    def audit_real(self, ctx, snapshots, horizon, gen):
        """The real rollouts: in batches of the envs' size, each env restores one start's snapshot and the batch is
        played in lockstep with the greedy action of the acting loop (encoder on the real frames, TSSM step) until each
        episode ends -> {"r" [N, H, 4] (sparse, standard, curiosity, death penalty), "steps" [N], "died" [N],
        "completed" [N], "obs" (encoder inputs of every step, [N, H, ...])}."""
        c, dev, N, H = self.cfg, self.device, len(snapshots), int(horizon)
        envs = self._audit_envs()
        E = len(envs)
        r = np.zeros((N, H, 4), dtype=np.float32)
        steps, died, completed = np.full(N, H, dtype=np.int64), np.zeros(N, dtype=bool), np.zeros(N, dtype=bool)
        per_step = [[None] * N for _ in range(H)]                          # observation of start i after step k
        for b0 in range(0, N, E):
            sel = list(range(b0, min(b0 + E, N)))
            n = len(sel)
            envs.restore([snapshots[i] for i in sel] + [None] * (E - n))
            idx_t = torch.as_tensor(sel, device=dev)
            cache = self.tssm.make_cache(n, dev, self.cache_dtype)
            self.tssm.prefill(cache, [(k[idx_t], v[idx_t]) for k, v in ctx["kv"]], torch.arange(n, device=dev),
                              ctx["t0"][idx_t])
            s = ctx["start"][idx_t]
            alive, last = [True] * n, [None] * n
            for k in range(H):
                with self._amp():
                    action = self._audit_action(s)
                    h = self.tssm.forward_step(s[:, self.tssm_dim:], action, cache).float()
                act = action.argmax(-1).cpu().numpy()
                results = envs.step(np.concatenate([act, np.zeros(E - n, dtype=act.dtype)]),
                                    active=alive + [False] * (E - n))
                for j, i in enumerate(sel):
                    if not alive[j]:
                        continue
                    o, _, terminated, truncated, info = results[j]
                    death = bool(terminated and info.get("died", True))
                    r[i, k] = (info.get("sparse_reward", 0.0), info.get("standard_reward", 0.0),
                               info.get("curiosity", 0.0), c.death_penalty if death else 0.0)
                    completed[i] |= info.get("sparse_reward", 0.0) > 0
                    last[j] = o
                    if terminated or truncated:
                        alive[j], steps[i], died[i] = False, k + 1, death
                for j, i in enumerate(sel):
                    per_step[k][i] = last[j]
                if not any(alive):
                    for kk in range(k + 1, H):                             # pad: never compared (steps)
                        for j, i in enumerate(sel):
                            per_step[kk][i] = last[j]
                    break
                obs_t = obs_to_tensors(last, dev)
                s = self.full_state(h, self._audit_encode(obs_t, gen))
        seen = []
        for k in range(H):
            o = obs_to_tensors(per_step[k], dev)
            o["blocks"] = torch.from_numpy(np.stack([x["blocks"] for x in per_step[k]])).long().to(dev)
            seen.append(o)
        obs = {key: torch.stack([o[key] for o in seen], dim=1) for key in seen[0]}
        return {"r": torch.as_tensor(r), "steps": steps, "died": died, "completed": completed, "obs": obs}

    @torch.no_grad()
    def _audit_compare(self, dream, real, ctx):
        """Dream vs real frames and labels -> {"frame_err" / "sprite_err" [N, H + 1] (RMSE of the decoded frame vs the
        real one, all pixels / sprite tiles; step 0 = the start frame's reconstruction), "level_hit" / "tier_hit" /
        "grid_hit" [N, H]}."""
        obs, frames = real["obs"], dream["frames"]
        N, H = frames.shape[0], frames.shape[1] - 1
        ctx_obs = ctx["obs0"]                                              # the start frames
        screens = torch.cat([ctx_obs["screen"][:, None], obs["screen"]], dim=1)
        palettes = torch.cat([ctx_obs["palette"][:, None], obs["palette"]], dim=1)
        blocks = torch.cat([ctx_obs["blocks"][:, None], obs["blocks"]], dim=1)
        rgb = frame_rgb(screens.flatten(0, 1), palettes.flatten(0, 1)).view(N, H + 1, 3, IMG_H, IMG_W)
        sprite = sprite_pixel_mask(blocks.flatten(0, 1)).view(N, H + 1, IMG_H, IMG_W)
        f, rr, sp = frames.flatten(0, 1), rgb.flatten(0, 1), sprite.flatten(0, 1)
        frame_err = rmse(f, rr).view(N, H + 1)
        sprite_err = rmse(f, rr, sp).view(N, H + 1)
        tier = ram_fields(obs["ram"])["tier"]
        return {"frame_err": frame_err, "sprite_err": sprite_err,
                "level_hit": (dream["level"] == obs["map_id"]).float(), "tier_hit": (dream["tier"] == tier).float(),
                "grid_hit": (dream["grid"].float() == obs["grid"].float()).float().mean(-1)}

    # --- Acting in the game -----------------------------------------------------------------------------------------
    def new_act_context(self, n):
        """Per-env acting state: TSSM cache and h (the static buffers of the acting CUDA graph when it is used)."""
        g = self._act_graph_for(n)
        if g is not None:
            g.reset()
            return {"cache": g.cache, "h": g.h, "graph": g}
        return {"cache": self.tssm.make_cache(n, self.device, self.cache_dtype),
                "h": torch.zeros(n, self.tssm_dim, device=self.device)}

    def _act_graph_for(self, n):
        """CUDA graph of the acting step for n envs, or None to act eagerly (CPU, disabled or capture failed)."""
        if not self._use_act_graph or self.device.type != "cuda":
            return None
        g = self._act_graphs.get(n)
        if g is None:
            try:
                t = time.perf_counter()
                g = GraphedActStep(self, n)
            except Exception as e:                                         # never let the graph stop training
                print(f"[!] acting CUDA graph disabled ({type(e).__name__}: {e}); acting runs eagerly.")
                self._use_act_graph = False
                return None
            self._act_graphs[n] = g
            print(f"\033[90m    [setup] acting: CUDA graph of one acting step captured and checked for {n} envs in "
                  f"{time.perf_counter() - t:.1f}s\033[0m")
        return g

    @staticmethod
    def reset_act_rows(ctx, rows):
        """Clears the acting state of envs that start a new episode."""
        ctx["cache"].reset_rows(rows)
        ctx["h"][rows] = 0.0

    @torch.no_grad()
    def act(self, observations, ctx, epsilon=0.0, greedy=False):
        """Encode observations, sample actions from [h_t, z_t], advance the TSSM -> one-hot actions [N, A]."""
        g = ctx.get("graph")
        if g is not None and epsilon == 0.0 and not greedy:
            return g(observations)
        obs = obs_to_tensors(observations, self.device)
        with self._amp():
            z, _ = self.tokenizer.encode(obs)
            z = z.float()
            action, _, _ = self.actor(self.full_state(ctx["h"], z), greedy=greedy, unimix=self.cfg.play_unimix)
            if epsilon > 0.0:
                override = torch.rand(action.shape[0], device=self.device) < epsilon
                if override.any():
                    ra = torch.randint(0, self.action_dim, (int(override.sum().item()),), device=self.device)
                    action[override] = F.one_hot(ra, self.action_dim).float()
            h = self.tssm.forward_step(z, action, ctx["cache"]).float()
            if g is not None:
                ctx["h"].copy_(h)                                          # the graph's static h
            else:
                ctx["h"] = h
        return action

    @torch.no_grad()
    def play(self, episodes_per_env=1, epsilon=None, snapshots=False):
        """Plays episodes in every env, stores them in the buffer -> (mean reward, mean curiosity). snapshots: every
        audit_snapshot_every steps each env also returns its complete state, kept (labelled) for the next dream
        audit."""
        epsilon = self.cfg.play_epsilon if epsilon is None else epsilon
        num_envs = len(self.envs)
        pool = self.audit_snapshots
        snap_on = bool(snapshots) and pool.every > 0
        if snap_on:
            pool.clear()
        nonplayable, skipped = 0, 0                        # steps on the map / transition screens; skipped by the env
        episodes_completed = [0] * num_envs
        scores, curiosity_scores, lengths, deaths, timeouts = [], [], [], 0, 0
        current_rewards = [0.0] * num_envs
        current_curiosities = [0.0] * num_envs
        local_buffers = [[] for _ in range(num_envs)]
        level_done = [False] * num_envs                    # level_end_terminal: the next frame is stored with cont 0
        maps_visited = [set() for _ in range(num_envs)]
        progress = [(-1, 0)] * num_envs                    # furthest (main level, x tile) of the running episode
        episode_progress = []

        def advance(i, coord):
            """Tracks the furthest main-level position of env i."""
            lid, x = int(coord[0]), int(coord[1])
            if lid < len(MAIN_LEVELS) and (lid, x) > progress[i]:
                progress[i] = (lid, x)

        ctx = self.new_act_context(num_envs)
        observations = []
        for i, (obs, info) in enumerate(self.envs.reset()):
            observations.append(obs)
            maps_visited[i].add(info["map_id"])
            advance(i, info["coord"])

        t_model = t_env = t_buffer = 0.0
        t_play = time.perf_counter()
        while min(episodes_completed) < episodes_per_env:
            t0 = time.perf_counter()
            action_idxs = torch.argmax(self.act(observations, ctx, epsilon=epsilon), dim=-1).cpu().numpy()
            t1 = time.perf_counter()
            t_model += t1 - t0

            active = [episodes_completed[i] < episodes_per_env for i in range(num_envs)]
            if snap_on:                                    # the post-step state = row len + 1 of the episode
                flags = [active[i] and pool.due(len(local_buffers[i]) + 1) for i in range(num_envs)]
                results = self.envs.step(action_idxs, active=active, snapshot=flags)
            else:
                results = self.envs.step(action_idxs, active=active)
            t_env += time.perf_counter() - t1
            envs_to_reset = []
            for i in range(num_envs):
                if not active[i]:
                    continue
                next_obs, reward, terminated, truncated, info = results[i]
                if self.verbose:
                    for msg in info.get("logs", ()):
                        print(f"    [Env {i+1}] {msg}")
                self.total_num_steps += 1
                nonplayable += int(not info.get("playable", True))
                skipped += int(info.get("skipped", 0))
                if "snapshot" in info:
                    pool.add(i, len(local_buffers[i]) + 1, info.pop("snapshot"))
                current_rewards[i] += reward
                cur_map = info["coord"][0]
                if cur_map not in maps_visited[i] and cur_map in MAP_NAMES and self.verbose:
                    print(f"    [Env {i+1}] \033[1;96mEntered {MAP_NAMES[cur_map]} for the first time\033[0m")
                maps_visited[i].add(cur_map)
                advance(i, info["coord"])
                curiosity = info.get("curiosity", 0.0)
                current_curiosities[i] += curiosity
                local_buffers[i].append((observations[i], int(action_idxs[i]), info.get("sparse_reward", 0.0),
                                         info.get("standard_reward", 0.0), curiosity,
                                         0.0 if level_done[i] and self.cfg.level_end_terminal else 1.0))
                level_done[i] = info.get("sparse_reward", 0.0) > 0.0
                observations[i] = next_obs
                if terminated or truncated:
                    if terminated:                                         # post-death frame: its own cont = 0 row
                        local_buffers[i].append((next_obs, 0, 0.0, 0.0, 0.0, 0.0))
                        deaths += int(info.get("died", True))
                    timeouts += int(info.get("timeout", False))            # truncated: no cont = 0 row
                    if self.verbose:
                        print(f"    [Env {i+1}] Episode done | Reward: {current_rewards[i]:.2f} | "
                              f"Curiosity: {current_curiosities[i]:.3f} | Steps: {info['steps']} | "
                              f"Maps: {sorted(maps_visited[i])}")
                    tb = time.perf_counter()
                    start_row, episode_id = self.buffer.index, self.buffer._episode_counter
                    self.buffer.add_episode(local_buffers[i])
                    if snap_on:
                        pool.close_episode(i, start_row, episode_id, len(local_buffers[i]),
                                           terminated and info.get("died", True), self.cfg.death_dream_window)
                    t_buffer += time.perf_counter() - tb
                    local_buffers[i] = []
                    level_done[i] = False
                    lengths.append(info["steps"])
                    scores.append(current_rewards[i])
                    curiosity_scores.append(current_curiosities[i])
                    episode_progress.append(progress[i])
                    progress[i] = (-1, 0)
                    self.total_num_episodes += 1
                    episodes_completed[i] += 1
                    maps_visited[i] = set()
                    if episodes_completed[i] < episodes_per_env:
                        envs_to_reset.append(i)
                        current_rewards[i] = current_curiosities[i] = 0.0
                        self.reset_act_rows(ctx, i)
            tr = time.perf_counter()
            for i in envs_to_reset:
                observations[i], info = self.envs.reset_one(i)
                maps_visited[i].add(info["map_id"])
                advance(i, info["coord"])
            t_env += time.perf_counter() - tr

        for i in range(num_envs):
            if local_buffers[i]:
                self.buffer.add_episode(local_buffers[i])
        if snap_on:
            pool.drop_open()

        if lengths:
            ls = sorted(lengths)
            n = len(ls)
            median = ls[n // 2] if n % 2 else (ls[n // 2 - 1] + ls[n // 2]) / 2
            short = sum(1 for x in ls if x < self.cfg.seq_len)
            print(f"\033[90m    [collect] {sum(ls):,} env steps in {n} episodes | length median {median:.0f} / "
                  f"mean {sum(ls) / n:.0f} / min {ls[0]} / max {ls[-1]} | deaths {deaths} | time-outs {timeouts} | "
                  f"{short} shorter than a training window ({self.cfg.seq_len})\033[0m")
            far_lvl, far_x = max(episode_progress) if episode_progress else (-1, 0)
            self.last_play_stats = {
                "deaths": deaths / n, "timeouts": timeouts / n, "episode_length": sum(ls) / n, "short_episodes": short / n, "steps": sum(ls),
                "t_play": time.perf_counter() - t_play, "t_model": t_model, "t_env": t_env, "t_buffer": t_buffer,
                "best_reward": max(scores), "best_curiosity": max(curiosity_scores),
                "furthest_level": far_lvl, "furthest_x": far_x,
                "nonplayable_steps": nonplayable / max(sum(ls), 1), "skipped_steps": skipped,
                "death_penalty": self.cfg.death_penalty * deaths / n,
                "audit_snapshots": len(pool) if snap_on else 0,
                "mean_x": sum(x for _, x in episode_progress) / max(len(episode_progress), 1)}

        avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
        avg_cur = round(sum(curiosity_scores) / len(curiosity_scores), 4) if curiosity_scores else 0.0
        return avg_score, avg_cur


class GraphedActStep:
    """agent.act for n envs (epsilon 0, sampled actions) as one CUDA graph: encoder posterior -> z sample -> actor ->
    action sample -> TSSM step on a static cache. The eager step launches ~500 small kernels (a few ms on Windows);
    the graph replays them in one call. The Gumbel noise of both samples is drawn outside the graph (same
    distributions as the eager path: Gumbel-max of the posterior / of log pi), so the graph itself is deterministic
    and is checked against the eager computation at capture."""

    def __init__(self, agent, n, warmup=3):
        from .game import PALETTE_SIZE, SCREEN_H, SCREEN_W
        from .observations import RAM_DIM
        self.agent, self.n = agent, n
        dev = self.device = agent.device
        self.cache = agent.tssm.make_cache(n, dev, agent.cache_dtype)
        self.h = torch.zeros(n, agent.tssm_dim, device=dev)
        spec = {"screen": ((n, SCREEN_H, SCREEN_W), torch.uint8), "palette": ((n, PALETTE_SIZE, 3), torch.uint8),
                "grid": ((n, GRID_DIM), torch.float32), "map_id": ((n,), torch.long), "ram": ((n, RAM_DIM), torch.uint8)}
        self.obs = {k: torch.zeros(sh, dtype=dt, device=dev) for k, (sh, dt) in spec.items()}
        self.staging = {k: torch.zeros(sh, dtype=dt, pin_memory=True) for k, (sh, dt) in spec.items()}
        self.gz = torch.zeros(n, agent.n_latents, agent.classes, device=dev)
        self.ga = torch.zeros(n, agent.action_dim, device=dev)
        self.graph = None
        self._capture(warmup)

    @torch.no_grad()
    def _body(self):
        ag = self.agent
        with eager_modules(), torch.autocast(self.device.type, dtype=ag.amp_dtype or torch.bfloat16,
                                             enabled=ag.amp_dtype is not None, cache_enabled=False):
            logp = ag.tokenizer.posterior(self.obs)
            z = torch.zeros_like(logp).scatter_(-1, (logp + self.gz).argmax(-1, keepdim=True), 1.0).flatten(1)
            probs = ag.actor.probs(ag.full_state(self.h, z), unimix=ag.cfg.play_unimix)
            a = torch.zeros_like(probs).scatter_(-1, (probs.clamp_min(1e-12).log() + self.ga).argmax(-1, keepdim=True),
                                                 1.0)
            self.h.copy_(ag.tssm.forward_step(z, a, self.cache).float())
        return a

    def _noise(self):
        self.gz.copy_(gumbel_like(self.gz))
        self.ga.copy_(gumbel_like(self.ga))

    def _randomize(self):
        g = self.obs
        g["screen"].random_(0, 32)
        g["palette"].random_(0, 256)
        g["grid"].copy_((torch.rand_like(g["grid"]) < 0.5).float())
        g["ram"].zero_()
        self._noise()

    def _capture(self, warmup):
        """Warms up, captures the graph and checks it against the eager step."""
        snap, h0 = self.cache.snapshot(), self.h.clone()

        def restore():
            self.cache.restore(snap)
            self.h.copy_(h0)
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                self._randomize()
                self._body()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        restore()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            self.a_out = self._body()
        restore()
        self._randomize()
        graph.replay()
        a_g, h_g = self.a_out.clone(), self.h.clone()
        restore()
        a_e = self._body()
        err = (self.h - h_g).abs().max().item()
        restore()
        tol = 5e-2 if self.agent.amp_dtype is not None else 1e-3
        if not (bool((a_e == a_g).all()) and err <= tol):
            raise RuntimeError(f"graph replay differs from the eager step (h error {err:.3g}, tolerance {tol})")
        self.graph = graph

    def reset(self):
        """Empties every row (a new play round)."""
        self.cache.reset_rows(slice(None))
        self.h.zero_()

    def __call__(self, observations):
        """Env observations -> one-hot actions [n, A] (a static tensor: read it before the next call)."""
        st = self.staging
        st["screen"].numpy()[:] = np.stack([o["screen"] for o in observations])
        st["palette"].numpy()[:] = np.stack([o["palette"] for o in observations])
        st["grid"].numpy()[:] = np.stack([o["grid"] for o in observations])
        st["map_id"].numpy()[:] = [int(o["map_id"]) for o in observations]
        st["ram"].numpy()[:] = np.stack([obs_ram(o) for o in observations])
        for k, v in st.items():
            self.obs[k].copy_(v, non_blocking=True)
        self._noise()
        self.graph.replay()
        return self.a_out
