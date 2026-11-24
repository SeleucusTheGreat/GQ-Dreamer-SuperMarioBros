"""Dream audit (diagnostic only): plays the same real states forward in the dream and in the real
game with the same greedy policy and compares returns per reward stream, death and level-completion rates, the
decoded frames against the real ones, the level / tier read from the dreamed latents and the reward heads' error."""
import math

import numpy as np
import torch

AUDIT_STREAMS = ("sparse", "standard", "curiosity", "penalty")   # penalty: the death penalty (change 7)
AUDIT_KINDS = ("uniform", "pre_death")
PRE_DEATH_SHARE = 0.3                                           # share of pre-death starts
FRAME_STEPS = (1, 5, 10, 25)                                    # decoded-frame error at these rollout steps
HEAD_STEPS = 5                                                  # reward-head error over the first rollout steps
COMPLETION_THRESHOLD = 25.0                                     # a sparse-reward prediction above this = a completion
_KIND_LABELS = {"uniform": "uniform", "pre_death": "pre-death starts"}

# metrics CSV columns (trainer.Trainer): (header, source, key); empty in a round without an audit
AUDIT_COLUMNS = [
    ("auditStarts", "audit", "starts"), ("auditSeconds", "audit", "seconds"),
    ("auditStartsPreDeath", "audit", "starts_pre_death"),
    *[(f"audit_{g}_{side}", "audit", f"{g}_{side}") for g in AUDIT_STREAMS for side in ("dream", "real", "gap")],
    *[(f"audit_{g}_pre_death_{side}", "audit", f"{g}_pre_death_{side}") for g in AUDIT_STREAMS
      for side in ("dream", "real", "gap")],
    ("audit_death_dream", "audit", "death_dream"), ("audit_death_real", "audit", "death_real"),
    ("audit_death_pre_death_dream", "audit", "death_pre_death_dream"),
    ("audit_death_pre_death_real", "audit", "death_pre_death_real"),
    ("audit_completion_dream", "audit", "completion_dream"), ("audit_completion_real", "audit", "completion_real"),
    ("audit_frame_err_0", "audit", "frame_err_0"), ("audit_sprite_err_0", "audit", "sprite_err_0"),
    *[(f"audit_frame_err_{s}", "audit", f"frame_err_{s}") for s in FRAME_STEPS],
    *[(f"audit_sprite_err_{s}", "audit", f"sprite_err_{s}") for s in FRAME_STEPS],
    ("audit_level_acc", "audit", "level_acc"), ("audit_tier_acc", "audit", "tier_acc"),
    ("audit_grid_acc", "audit", "grid_acc"),
    *[(f"audit_head_err_{g}", "audit", f"head_err_{g}") for g in AUDIT_STREAMS[:3]],
]


def audit_enabled(cfg):
    """True when the dream audit runs (cfg.audit on and cfg.audit_every > 0)."""
    return bool(cfg.audit) and int(cfg.audit_every) > 0


class SnapshotPool:
    """Env snapshots of the play just finished, one every `every` agent steps of an episode, labelled when it ends."""

    def __init__(self, every):
        self.every = max(int(every), 0)                 # 0: no snapshots (audit off)
        self.items = []                                 # labelled snapshots of finished episodes
        self._open = {}                                 # env index -> [(step, snapshot)] of its running episode

    def __len__(self):
        return len(self.items)

    def clear(self):
        """Forgets every snapshot (an audit uses the latest play only)."""
        self.items, self._open = [], {}

    def due(self, step):
        """True when the observation of episode step `step` (its row index) gets a snapshot."""
        return self.every > 0 and step > 0 and step % self.every == 0

    def add(self, env, step, snapshot):
        """Adds env `env`'s snapshot of the state whose observation is row `step` of its running episode."""
        self._open.setdefault(env, []).append((int(step), snapshot))

    def drop_open(self):
        """Forgets the snapshots of episodes that were not stored."""
        self._open = {}

    def close_episode(self, env, start_row, episode_id, length, died, death_window):
        """Labels the snapshots of env's stored episode (replay rows start_row.., id episode_id, `length` rows; with a
        death its last row is the death frame) as pre-death (the death death_window[0]..[1] steps later) or not."""
        lo, hi = death_window
        for t, snap in self._open.pop(env, []):
            if t >= length:
                continue
            pre = bool(died) and lo <= (length - 1) - t <= hi
            self.items.append({"snapshot": snap, "step": t, "row": int(start_row) + t, "episode_id": int(episode_id),
                               "pre_death": pre})

    def usable(self, buffer, context):
        """Labelled snapshots whose replay rows (the start and up to `context` earlier steps) still hold their
        episode."""
        out, n = [], len(buffer)
        for it in self.items:
            rows = [it["row"] % buffer.capacity, (it["row"] - min(it["step"], context)) % buffer.capacity]
            if all(r < n and int(buffer.episode_ids[r]) == it["episode_id"] for r in rows) \
                    and int(buffer.ep_steps[rows[0]]) == it["step"]:
                out.append(it)
        return out


def select_starts(items, n, rng):
    """Picks the starts -> [(index into items, kind)]: round(PRE_DEATH_SHARE x n) pre-death, the rest uniform (also
    filling a missing kind); never a snapshot twice. rng: a numpy Generator."""
    n = min(int(n), len(items))
    taken = np.zeros(len(items), dtype=bool)
    picks = []
    for kind, flag, want in (("pre_death", "pre_death", int(round(PRE_DEATH_SHARE * n))), ("uniform", None, n)):
        cands = [i for i, it in enumerate(items) if not taken[i] and (flag is None or it[flag])]
        k = min(want, n - len(picks), len(cands))
        if k <= 0:
            continue
        for i in rng.choice(np.asarray(cands, dtype=np.int64), size=k, replace=False):
            taken[i] = True
            picks.append((int(i), kind))
    return picks


def audit_metrics(dream, real, kinds, discount):
    """Audit metrics of N starts over H steps.

    dream: "r" [N, H, 4] per-stream rewards (sparse, standard, curiosity, death penalty), "alive" [N, H] calibrated
    p(alive) of each new state, "frame_err" / "sprite_err" [N, H + 1] (step 0: the start frame's reconstruction),
    "level_hit" / "tier_hit" / "grid_hit" [N, H] (dreamed latents vs the real game); real: "r" [N, H, 4], "steps" [N]
    (steps taken before the episode ended), "died" [N], "completed" [N]. Returns per stream the discounted return
    (dream weighted by its predicted survival, real over the steps taken) and the gap, the death rates, the completion
    rates, the frame errors at FRAME_STEPS, the grounding accuracies and the reward heads' error over HEAD_STEPS;
    overall and for pre-death starts."""
    rd, alive = (torch.as_tensor(dream[k]).detach().float().cpu() for k in ("r", "alive"))
    rr = torch.as_tensor(real["r"]).float().cpu()
    steps = torch.as_tensor(np.asarray(real["steps"]), dtype=torch.long)
    died = torch.as_tensor(np.asarray(real["died"]), dtype=torch.float32)
    completed = torch.as_tensor(np.asarray(real["completed"]), dtype=torch.float32)
    kinds = torch.as_tensor(np.asarray(kinds), dtype=torch.long)
    N, H, G = rd.shape
    k = torch.arange(H)
    disc = float(discount) ** k.double()
    surv = torch.cat([torch.ones(N, 1), torch.cumprod(alive, dim=1)[:, :-1]], dim=1).double()
    taken = (k[None] < steps[:, None]).double()
    ret_d = ((disc * surv)[..., None] * rd.double()).sum(1)                       # [N, G]
    ret_r = ((disc * taken)[..., None] * rr.double()).sum(1)
    death_d = 1.0 - alive.double().prod(1)
    comp_d = ((surv * (rd[..., 0] >= COMPLETION_THRESHOLD).double()).sum(1)).clamp(max=1.0)
    out = {"starts": float(N), "horizon": float(H)}
    for name, sel in (("", torch.ones(N, dtype=torch.bool)), ("_pre_death", kinds == AUDIT_KINDS.index("pre_death"))):
        if name:
            out[f"starts{name}"] = float(sel.sum())
        if not bool(sel.any()):
            continue
        for gi, g in enumerate(AUDIT_STREAMS[:G]):
            d, r = float(ret_d[sel, gi].mean()), float(ret_r[sel, gi].mean())
            out.update({f"{g}{name}_dream": d, f"{g}{name}_real": r, f"{g}{name}_gap": d - r})
        out[f"death{name}_dream"] = float(death_d[sel].mean())
        out[f"death{name}_real"] = float(died[sel].mean())
    out["completion_dream"] = float(comp_d.mean())
    out["completion_real"] = float(completed.mean())
    fe, se = (torch.as_tensor(dream[k]).float().cpu() for k in ("frame_err", "sprite_err"))
    out["frame_err_0"], out["sprite_err_0"] = float(fe[:, 0].mean()), float(se[:, 0].nanmean())
    for s in FRAME_STEPS:
        running = steps >= s
        if s > H or not bool(running.any()):
            continue
        out[f"frame_err_{s}"] = float(fe[running, s].mean())
        v = se[running, s]
        out[f"sprite_err_{s}"] = float(v.nanmean()) if bool((~v.isnan()).any()) else float("nan")
    m = taken.float()
    for key in ("level", "tier", "grid"):
        hit = torch.as_tensor(dream[f"{key}_hit"]).float().cpu()
        out[f"{key}_acc"] = float((hit * m).sum() / m.sum().clamp(min=1.0))
    mh = taken[:, :min(HEAD_STEPS, H)]
    if float(mh.sum()) > 0:
        for gi, g in enumerate(AUDIT_STREAMS[:3]):
            err = (rd[:, :mh.shape[1], gi] - rr[:, :mh.shape[1], gi]).abs().double()
            out[f"head_err_{g}"] = float((err * mh).sum() / mh.sum())
    return out


def audit_lines(m):
    """The audit block of the round summary ([] without an audit)."""
    if not m or not m.get("starts"):
        return []

    def pct(v):
        return "–" if v is None or v != v else f"{100 * v:.0f}%"

    def val(v):
        return "–" if v is None or v != v else f"{v:+.2f}"

    def num(v, d=3):
        return "–" if v is None or v != v else f"{v:.{d}f}"

    H = int(m.get("horizon", FRAME_STEPS[-1]))
    lines = [f"DREAM AUDIT · {int(m['starts'])} real states ({int(m.get('starts_pre_death', 0))} pre-death), each "
             f"played {H} steps in the dream and in the game ({m.get('seconds', 0.0):.1f} s)",
             f"  {f'return over {H} steps':<24}{'dream':>8}  {'real':>8}  {'gap':>8}   gap = dream − real "
             f"(+ = the dream is too optimistic)"]
    for g in AUDIT_STREAMS:
        if f"{g}_dream" in m:
            pre = (f"   pre-death {val(m.get(f'{g}_pre_death_dream'))} / {val(m.get(f'{g}_pre_death_real'))}"
                   if f"{g}_pre_death_dream" in m else "")
            lines.append(f"    {g:<22}{val(m[f'{g}_dream']):>8}  {val(m.get(f'{g}_real')):>8}  "
                         f"{val(m[f'{g}_dream'] - m.get(f'{g}_real', float('nan'))):>8}{pre}")
    pre = (f"   pre-death {pct(m.get('death_pre_death_dream'))} / {pct(m.get('death_pre_death_real'))}"
           if "death_pre_death_dream" in m else "")
    lines.append(f"  {f'died within {H} steps':<24}{pct(m.get('death_dream')):>8}  {pct(m.get('death_real')):>8}"
                 f"{'':>10}{pre}")
    lines.append(f"  {'completed a level':<24}{pct(m.get('completion_dream')):>8}  {pct(m.get('completion_real')):>8}")
    errs = " · ".join(f"{s} {num(m.get(f'frame_err_{s}'))} ({num(m.get(f'sprite_err_{s}'))})" for s in FRAME_STEPS)
    lines.append(f"  {'frame RMSE (sprites)':<24}start {num(m.get('frame_err_0'))} "
                 f"({num(m.get('sprite_err_0'))}) · {errs}")
    lines.append(f"  {'dreamed latents':<24}level acc {num(m.get('level_acc'))} · tier acc {num(m.get('tier_acc'))} · "
                 f"grid acc {num(m.get('grid_acc'))} · head error (5 steps) sparse "
                 f"{num(m.get('head_err_sparse'))} / standard {num(m.get('head_err_standard'))} / curiosity "
                 f"{num(m.get('head_err_curiosity'))}")
    return lines


def rmse(a, b, mask=None):
    """Per-frame RMSE of images [N, 3, H, W] (mask [N, H, W]: only those pixels; NaN where a frame has none)."""
    sq = (a.float() - b.float()).square().mean(1)                                  # [N, H, W]
    if mask is None:
        return sq.flatten(1).mean(1).sqrt()
    m = mask.float()
    n = m.flatten(1).sum(1)
    v = (sq * m).flatten(1).sum(1) / n.clamp(min=1.0)
    return torch.where(n > 0, v.sqrt(), torch.full_like(v, math.nan))
