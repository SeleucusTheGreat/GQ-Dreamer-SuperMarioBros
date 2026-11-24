"""Episode-aware replay buffer (padded windows, death / reward / recent / surprise sampling) and a batch prefetcher."""
import json
import os
import queue
import threading

import numpy as np
import torch

from .game import (BLOCK_COLS, BLOCK_ROWS, FRAMES_PER_STEP, LEVEL_TIMER_STEPS, PALETTE_SIZE, SCREEN_H, SCREEN_W,
                   level_steps_of)
from .observations import GRID_BYTES, RAM_DIM, obs_ram


class ReplayBuffer:
    """Ring buffer of whole episodes; row t = observation t, its action, the rewards of t -> t+1, continue flag of t."""
    FORMAT = "smb_fusion_v0"                                            # file-format id
    FILE_MAGIC = b"SMBBUF01"
    FIELDS = {                                                          # name: (shape, dtype)
        "screens": ((SCREEN_H, SCREEN_W), torch.uint8),
        "palettes": ((PALETTE_SIZE, 3), torch.uint8),
        "blocks": ((BLOCK_ROWS, BLOCK_COLS), torch.uint8),
        "grids": ((GRID_BYTES,), torch.uint8),
        "map_ids": ((), torch.uint8),
        "rams": ((RAM_DIM,), torch.uint8),
        "actions": ((), torch.uint8),
        "sparse_rewards": ((1,), torch.float32),
        "standard_rewards": ((1,), torch.float32),
        "curiosities": ((1,), torch.float32),
        "continues": ((1,), torch.float32),
    }
    META = ("episode_ids", "ep_steps", "ep_lens")
    _CHUNK = 2048                                                       # rows per file copy step
    _ALIGN = 64

    def __init__(self, capacity=500000, num_envs=1, frames_per_step=FRAMES_PER_STEP, death_sample_prob=0.0,
                 recent_sample_prob=0.0, reward_sample_prob=0.0, reward_threshold=50.0, loss_sample_prob=0.0,
                 loss_priority_alpha=1.0):
        self.capacity = capacity
        self.num_envs = num_envs
        self.frames_per_step = frames_per_step
        self.death_sample_prob = death_sample_prob
        self.recent_sample_prob = recent_sample_prob
        self.reward_sample_prob = reward_sample_prob
        self.reward_threshold = reward_threshold
        self.loss_sample_prob = loss_sample_prob                        # share of surprise-prioritised windows
        self.loss_priority_alpha = loss_priority_alpha

        for name, (shape, dtype) in self.FIELDS.items():
            setattr(self, name, torch.zeros((capacity, *shape), dtype=dtype))
        self.episode_ids = torch.full((capacity,), -1, dtype=torch.long)
        self.ep_steps = torch.zeros((capacity,), dtype=torch.int32)
        self.ep_lens = torch.zeros((capacity,), dtype=torch.int32)
        self.level_steps = torch.zeros((capacity,), dtype=torch.int32)  # derived from map ids, not saved
        self.priorities = torch.full((capacity,), -1.0, dtype=torch.float32)   # world-model KL, -1 = unseen
        self.default_priority = 1.0
        self.avg_episode_length = 0.0
        self.episode_len_ema_decay = 0.9
        self._episode_counter = 0
        self.index = 0
        self.full = False
        self._pools_T = None                                            # window length of the cached start pools

    def __len__(self):
        return self.capacity if self.full else self.index

    # --- Writing ----------------------------------------------------------------------------------------------------
    def add_episode(self, transitions):
        """Writes one episode: list of (obs, action, sparse_reward, standard_reward, curiosity, cont). One write per
        field for the whole episode."""
        L = len(transitions)
        if L == 0:
            return
        obs = [t[0] for t in transitions]
        maps = np.array([int(o["map_id"]) & 0xFF for o in obs], dtype=np.uint8)
        col = lambda j: torch.tensor([[float(t[j])] for t in transitions], dtype=torch.float32)   # noqa: E731
        fields = {
            "screens": torch.from_numpy(np.stack([np.asarray(o["screen"], dtype=np.uint8) for o in obs])),
            "palettes": torch.from_numpy(np.stack([np.asarray(o["palette"], dtype=np.uint8) for o in obs])),
            "blocks": torch.from_numpy(np.stack([np.asarray(o["blocks"], dtype=np.uint8) for o in obs])),
            "grids": torch.from_numpy(np.packbits(np.stack([np.asarray(o["grid"]) for o in obs]) > 0.5, axis=-1,
                                                  bitorder="little")),
            "map_ids": torch.from_numpy(maps),
            "rams": torch.from_numpy(np.stack([obs_ram(o) for o in obs])),
            "actions": torch.tensor([int(t[1]) for t in transitions], dtype=torch.uint8),
            "sparse_rewards": col(2), "standard_rewards": col(3), "curiosities": col(4), "continues": col(5),
        }
        rows = (self.index + torch.arange(L)) % self.capacity
        for name, v in fields.items():
            getattr(self, name)[rows] = v
        self.episode_ids[rows] = self._episode_counter
        self.ep_steps[rows] = torch.arange(L, dtype=torch.int32)
        self.ep_lens[rows] = L
        self.level_steps[rows] = torch.from_numpy(level_steps_of(maps))
        self.priorities[rows] = -1.0
        self.full = self.full or self.index + L >= self.capacity
        self.index = (self.index + L) % self.capacity
        self._episode_counter += 1
        d = self.episode_len_ema_decay
        prev = self.avg_episode_length
        self.avg_episode_length = float(L) if prev == 0.0 else d * prev + (1.0 - d) * L
        self._pools_T = None

    def update_priorities(self, rows, values):
        """Stores the world model's KL of these rows as their sampling priority."""
        if rows.numel() == 0:
            return
        values = values.float().clamp_min(0.0)
        self.priorities[rows] = values
        self.default_priority = 0.99 * self.default_priority + 0.01 * float(values.mean())

    # --- Sampling ---------------------------------------------------------------------------------------------------
    def _recent_span(self):
        """Rows written during roughly the last round of play."""
        return int(round(self.avg_episode_length * self.num_envs))

    def _build_pools(self, T):
        """Caches the window-start pools for windows of length T."""
        N = len(self)
        pos = torch.arange(N)
        steps, lens = self.ep_steps[:N].long(), self.ep_lens[:N].long()
        remaining = lens - steps
        starts = pos[(remaining >= T) | ((steps == 0) & (lens < T))]   # full windows + starts of short episodes
        deaths = pos[(self.continues[:N, 0] <= 0.0) & (steps == lens - 1)]   # terminal row of its episode (no flagpoles)
        back = torch.clamp(self.ep_steps[deaths].long(), max=T - 1)     # how far before the death a window may start
        ok = (self.episode_ids[(deaths - back) % self.capacity] == self.episode_ids[deaths]) & (back > 0)
        rewards = pos[self.sparse_rewards[:N, 0] >= self.reward_threshold]
        span = min(self._recent_span(), N)
        recent = starts[((self.index - 1 - starts) % self.capacity) < span] if span > 0 else starts[:0]
        self._pools = {"starts": starts, "deaths": deaths[ok], "death_back": back[ok], "rewards": rewards,
                       "recent": recent}
        self._pools_T = T

    def _death_starts(self, n):
        """Starts of n windows that contain a death, at a uniform index 1 .. T-1 of the window."""
        pools = self._pools
        j = torch.randint(0, pools["deaths"].numel(), (n,))
        db = pools["death_back"][j]
        back = 1 + (torch.rand(n) * db).long().clamp(max=db - 1)
        return (pools["deaths"][j] - back) % self.capacity

    def _anchored_starts(self, r, T):
        """Window starts that put rows r at a uniform index of their window (same episode)."""
        off = (torch.rand(r.shape[0]) * (torch.clamp(self.ep_steps[r].long(), max=T - 1) + 1)).long()
        s = (r - off) % self.capacity
        return torch.where(self.episode_ids[s] == self.episode_ids[r], s, r)

    def _priority_rows(self, n):
        """n rows drawn with probability ~ priority ** alpha (unseen rows: the running mean)."""
        p = self.priorities[:len(self)]
        w = torch.where(p < 0, torch.full_like(p, self.default_priority), p).clamp_min(1e-3)
        if self.loss_priority_alpha != 1.0:
            w = w ** self.loss_priority_alpha
        return torch.multinomial(w, n, replacement=True)

    def _vlen(self, starts, T):
        """Real (unpadded) length of each window."""
        rem = (self.ep_lens[starts] - self.ep_steps[starts]).long()
        return torch.clamp(rem, 1, T)

    def sample(self, batch_size, seq_len):
        """Batch of windows {field: [B, T, ...]} + level_steps, valid mask and buffer rows; None if empty."""
        T = seq_len
        if len(self) == 0:
            return None
        if self._pools_T != T:
            self._build_pools(T)
        pools = self._pools
        if pools["starts"].numel() == 0 and pools["deaths"].numel() == 0:
            return None

        def pick(pool, n):
            return pool[torch.randint(0, pool.numel(), (n,))]

        starts = pick(pools["starts"], batch_size) if pools["starts"].numel() > 0 else self._death_starts(batch_size)
        u = torch.rand(batch_size)
        p_d = self.death_sample_prob if pools["deaths"].numel() > 0 else 0.0
        p_r = self.reward_sample_prob if pools["rewards"].numel() > 0 else 0.0
        p_c = self.recent_sample_prob if pools["recent"].numel() > 0 else 0.0
        p_l = self.loss_sample_prob
        death_m = u < p_d
        reward_m = (u >= p_d) & (u < p_d + p_r)
        recent_m = (u >= p_d + p_r) & (u < p_d + p_r + p_c)
        loss_m = (u >= p_d + p_r + p_c) & (u < p_d + p_r + p_c + p_l)
        if death_m.any():
            starts[death_m] = self._death_starts(int(death_m.sum()))
        if recent_m.any():
            starts[recent_m] = pick(pools["recent"], int(recent_m.sum()))
        if reward_m.any():
            starts[reward_m] = self._anchored_starts(pick(pools["rewards"], int(reward_m.sum())), T)
        if loss_m.any():
            starts[loss_m] = self._anchored_starts(self._priority_rows(int(loss_m.sum())), T)

        vlen = self._vlen(starts, T)
        offs = torch.arange(T)[None]
        rows = (starts[:, None] + torch.minimum(offs, vlen[:, None] - 1)) % self.capacity   # pads repeat the last row
        batch = {name: getattr(self, name)[rows] for name in self.FIELDS}
        batch["level_steps"] = self.level_steps[rows]
        batch["valid"] = (offs < vlen[:, None]).float()
        batch["episode_end"] = self.ep_steps[rows] == self.ep_lens[rows] - 1   # last row of its episode
        batch["index"] = rows
        return batch

    def sample_frames(self, n):
        """n random stored frames (encoder-drift probe)."""
        idx = torch.randint(0, len(self), (n,))
        out = {name: getattr(self, name)[idx] for name in ("screens", "palettes", "grids", "map_ids", "rams")}
        out["level_steps"] = self.level_steps[idx]
        return out

    # --- Persistence (JSON header + raw field bytes) ----------------------------------------------------------------
    def save(self, path):
        """Writes the buffer to one file (atomic replace)."""
        n = len(self)
        names = list(self.FIELDS) + list(self.META)
        arrays = {name: getattr(self, name)[:n].numpy() for name in names}
        fields, off = {}, 0
        for name, a in arrays.items():
            fields[name] = {"dtype": a.dtype.str, "shape": list(a.shape), "offset": off}
            off += -(-a.nbytes // self._ALIGN) * self._ALIGN
        header = json.dumps({"format": self.FORMAT, "n": n, "index": self.index, "full": self.full,
                             "avg_episode_length": self.avg_episode_length, "frames_per_step": self.frames_per_step,
                             "timeouts_truncated": True, "fields": fields}).encode()
        data_start = -(-(len(self.FILE_MAGIC) + 8 + len(header)) // self._ALIGN) * self._ALIGN
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(self.FILE_MAGIC + np.uint64(len(header)).tobytes() + header)
            for name, a in arrays.items():
                f.seek(data_start + fields[name]["offset"])
                for r in range(0, max(n, 1), self._CHUNK):
                    f.write(np.ascontiguousarray(a[r:r + self._CHUNK]).data)
            f.truncate(data_start + off)
        os.replace(tmp, path)

    @classmethod
    def open_file(cls, path):
        """-> (header dict, {field: read-only memory-mapped array}) of a saved buffer."""
        with open(path, "rb") as f:
            if f.read(len(cls.FILE_MAGIC)) != cls.FILE_MAGIC:
                raise ValueError(f"{path} is not a replay buffer file")
            hlen = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
            meta = json.loads(f.read(hlen))
        data_start = -(-(len(cls.FILE_MAGIC) + 8 + hlen) // cls._ALIGN) * cls._ALIGN
        arrays = {}
        for name, fd in meta["fields"].items():
            shape = tuple(fd["shape"])
            arrays[name] = np.memmap(path, dtype=np.dtype(fd["dtype"]), mode="r",
                                     offset=data_start + fd["offset"], shape=shape) if shape[0] > 0 \
                else np.zeros(shape, dtype=np.dtype(fd["dtype"]))
        return meta, arrays

    def load(self, path):
        """Restores the newest `capacity` rows of a saved buffer (oldest first)."""
        names = list(self.FIELDS) + list(self.META)
        try:
            meta, src = self.open_file(path)
        except (OSError, ValueError) as e:
            print(f"[!] could not read replay buffer {path} ({e}); ignoring it.")
            return
        if meta.get("format") != self.FORMAT or any(name not in src for name in names):
            print(f"[!] {path} is not a {self.FORMAT} buffer; ignoring it.")
            return
        saved_fps = meta.get("frames_per_step")
        if saved_fps != self.frames_per_step:
            print(f"[!] {path} was recorded at {saved_fps} frames/step, the envs run at {self.frames_per_step}: "
                  f"ignoring it.")
            return
        src = {name: src[name] for name in names}
        n_saved = int(meta["n"])
        first = int(meta.get("index") or 0) % max(n_saved, 1) if meta.get("full") else 0
        n = min(n_saved, self.capacity)
        order = (first + np.arange(n_saved - n, n_saved)) % max(n_saved, 1)
        for a in range(0, n, self._CHUNK):
            rows = order[a:a + self._CHUNK]
            contiguous = rows[-1] - rows[0] == len(rows) - 1
            for name, arr in src.items():
                chunk = arr[rows[0]:rows[-1] + 1] if contiguous else arr[rows]
                getattr(self, name)[a:a + len(rows)] = torch.from_numpy(np.array(chunk))
        del src
        self.full = n == self.capacity
        self.index = n % self.capacity
        self._episode_counter = int(self.episode_ids[:n].max().item()) + 1 if n > 0 else 0
        self._recompute_level_steps(n)
        if not meta.get("timeouts_truncated"):                          # buffers that stored time-outs as deaths
            k = self._truncate_timeouts(n)
            print(f"\033[90m    [setup] replay buffer: {k} time-out endings relabelled as truncations "
                  f"(continue 0 -> 1)\033[0m")
        self.avg_episode_length = float(meta.get("avg_episode_length") or 0.0)
        self._pools_T = None
        print(f"\033[90m    [setup] replay buffer: {n:,} transitions restored "
              f"({100.0 * n / self.capacity:.0f}% full)\033[0m")

    def _recompute_level_steps(self, n):
        """Rebuilds level_steps of rows 0..n-1 after a load (a truncated oldest episode keeps its offset)."""
        maps = self.map_ids[:n].numpy()
        steps = self.ep_steps[:n].numpy().astype(np.int64)
        bounds = np.flatnonzero(steps == 0).tolist()
        if not bounds or bounds[0] != 0:
            bounds = [0] + bounds
        bounds.append(n)
        out = np.zeros(n, dtype=np.int32)
        for a, b in zip(bounds[:-1], bounds[1:]):
            ls = level_steps_of(maps[a:b])
            base = int(steps[a])
            if base:
                reset = np.flatnonzero(ls[1:] == 0)
                first_reset = int(reset[0]) + 1 if reset.size else len(ls)
                ls[:first_reset] += base
            out[a:b] = ls
        self.level_steps[:n] = torch.from_numpy(out)

    def _truncate_timeouts(self, n):
        """Relabels level time-outs stored as deaths (cont = 0 at least LEVEL_TIMER_STEPS steps into a level) as
        truncations (cont = 1) -> number of relabelled episodes."""
        last = self.ep_steps[:n] == self.ep_lens[:n] - 1
        rows = torch.nonzero(last & (self.continues[:n, 0] <= 0.0)
                             & (self.level_steps[:n] >= LEVEL_TIMER_STEPS)).squeeze(1)
        self.continues[rows, 0] = 1.0
        return int(rows.numel())

    def print_diagnostics(self):
        """Prints fill level and sampling statistics."""
        n = len(self)
        deaths = int(((self.continues[:n, 0] <= 0.0) & (self.ep_steps[:n] == self.ep_lens[:n] - 1)).sum())
        events = int((self.sparse_rewards[:n, 0] >= self.reward_threshold).sum())
        episodes = int(((self.ep_steps[:n] == 0) & (self.continues[:n, 0] > 0)).sum())
        print("\n" + "=" * 50)
        print("          REPLAY BUFFER DIAGNOSTICS")
        print("=" * 50)
        print(f"  Buffer Fill : {n:,} / {self.capacity:,} ({100.0 * n / self.capacity:.2f}%)")
        print(f"  Episodes : ~{episodes:,} stored | avg length {self.avg_episode_length:.0f}")
        print(f"  Recent Sampling : p={self.recent_sample_prob} | window {self._recent_span():,} steps")
        print(f"  Death Sampling : p={self.death_sample_prob} | {deaths:,} deaths (level completions and time-outs excluded), "
              f"each at a random index of its window")
        print(f"  Event Sampling : p={self.reward_sample_prob} | {events} +{self.reward_threshold:.0f} events")
        if self.loss_sample_prob > 0:
            seen = self.priorities[:n] >= 0
            print(f"  Surprise Sampling : p={self.loss_sample_prob} | priority = world-model KL of the frame "
                  f"(alpha {self.loss_priority_alpha}) | {int(seen.sum()):,} frames scored, mean "
                  f"{self.default_priority:.2f} nats")
        print("=" * 50 + "\n")


def with_batch_extras(cpu):
    """CPU batch -> (batch + real-frame gather indices, CPU-side extras). Every row's real frames come first and its
    pads repeat the last real one: real_idx lists the real frames of the flattened [B, T] batch, real_gather maps
    every position to its frame in real_idx (skip_pad_frames). The extras stay on the CPU: the number of real frames
    with t >= 1 and the buffer rows / valid mask for the surprise priorities (no host sync on the GPU side)."""
    valid = cpu["valid"]
    B, T = valid.shape
    vlen = valid.sum(1).long().clamp(min=1)
    t = torch.arange(T)
    real = (t[None] < vlen[:, None]).flatten().nonzero().squeeze(1)
    pos = torch.full((B * T,), -1, dtype=torch.long)
    pos[real] = torch.arange(real.numel())
    src = (torch.arange(B)[:, None] * T + torch.minimum(t[None], vlen[:, None] - 1)).flatten()
    extra = {"n_valid1": int(valid[:, 1:].sum()), "index_cpu": cpu["index"], "valid_cpu": valid}
    return {**cpu, "real_idx": real, "real_gather": pos[src]}, extra


class BatchPrefetcher:
    """Background thread that keeps the next replay batches ready in (pinned) CPU memory, or already on the GPU
    (device: copied on the thread's own CUDA stream while the current update runs)."""

    def __init__(self, buffer, batch_size, seq_len, pin, depth=2, device=None):
        self.batch_size, self.seq_len = batch_size, seq_len
        self.device = device
        self._queue = queue.Queue(maxsize=1 if device is not None else depth)     # GPU batches: one ahead
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(buffer, pin), daemon=True)
        self._thread.start()

    def _run(self, buffer, pin):
        """Producer loop; errors are passed to get()."""
        stream = torch.cuda.Stream(self.device) if self.device is not None else None
        while not self._stop.is_set():
            try:
                item = buffer.sample(self.batch_size, self.seq_len)
                if item is not None:
                    cpu, extra = with_batch_extras(item)
                    if pin:
                        cpu = {k: v.pin_memory() for k, v in cpu.items()}
                    if stream is not None:
                        with torch.cuda.stream(stream):
                            gpu = {k: v.to(self.device, non_blocking=True) for k, v in cpu.items()}
                            ev = torch.cuda.Event()
                            ev.record(stream)
                        item = {"__gpu__": gpu, "__event__": ev, "__cpu__": extra}
                    else:
                        item = {"__batch__": cpu, "__cpu__": extra}
            except Exception as e:
                item = e
            while not self._stop.is_set():
                try:
                    self._queue.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue
            if item is None or isinstance(item, Exception):
                return

    def get(self):
        """Next batch (re-raises producer errors)."""
        item = self._queue.get()
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        """Stops the thread and drops queued batches."""
        self._stop.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=5)
