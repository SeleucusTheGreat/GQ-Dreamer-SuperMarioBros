"""Death maps drawn while training. Every `death_map_interval` rounds the trainer copies what the death map needs from
its in-memory replay buffer (the episode columns of every row and the last frames of every episode: ~60 MB, well
under a second) and diagnostics/images/death_replay.py replays and draws it in a worker process with its own pool of
emulators, at below-normal priority, while training goes on. The image is diagnostics/images/death_map_G<update>.png,
the same one death_image.ipynb draws by hand from a saved buffer."""
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent                       # project root (holds gq_dreamer/)
DEATH_REPLAY_DIR = ROOT / "diagnostics" / "images"                  # death_replay.py
TAIL_ROWS = 12                       # frames kept per episode: its last rows (death frame and the frames shown before)
EPISODE_FIELDS = ("actions", "map_ids", "continues", "episode_ids", "ep_steps", "ep_lens")


def snapshot_buffer(buffer, tail=TAIL_ROWS):
    """The replay buffer as death_replay.BufferSnapshot reads it, oldest row first: dict with the episode columns of
    every row and the screens / palettes of the last `tail` rows of every episode (plain numpy, safe to pickle)."""
    import torch
    n = len(buffer)
    order = (buffer.index + torch.arange(n)) % buffer.capacity if buffer.full else torch.arange(n)
    fields = {k: getattr(buffer, k)[order].numpy() for k in EPISODE_FIELDS}
    frame_rows = np.flatnonzero(fields["ep_steps"].astype(np.int64) >= fields["ep_lens"].astype(np.int64) - tail)
    rows = order[torch.from_numpy(frame_rows)]
    return {"n": n, "frames_per_step": int(buffer.frames_per_step), "fields": fields, "frame_rows": frame_rows,
            "screens": buffer.screens[rows].numpy(), "palettes": buffer.palettes[rows].numpy()}


def _below_normal_priority():
    """Lowers this process's priority (inherited by the replay workers it starts) so play and training keep theirs."""
    try:
        if sys.platform == "win32":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(5)
    except Exception:                                                # priority is a nicety: never fail on it
        pass


def _draw(snap, update, config, workers):
    """Worker process: replays and draws the snapshot -> (image path, summary text, log)."""
    _below_normal_priority()
    if str(DEATH_REPLAY_DIR) not in sys.path:
        sys.path.insert(0, str(DEATH_REPLAY_DIR))
    import death_replay
    return death_replay.death_map_snapshot(snap, update, config=config, workers=workers)


class DeathMapJob:
    """One death map at a time, drawn by a worker process (spawned on first use, kept for the next maps)."""

    def __init__(self, config="gq", workers=None):
        self.config = config
        self.workers = workers                                       # replay processes; None = CPU count - 2 (max 8)
        self._pool = None
        self._future = None
        self.update = None                                           # gradient update of the map being drawn

    @property
    def busy(self):
        return self._future is not None and not self._future.done()

    def submit(self, buffer, update):
        """Copies the buffer and starts drawing its map -> False when the previous map is still being drawn."""
        if self.busy:
            return False
        if not (DEATH_REPLAY_DIR / "death_replay.py").exists():
            raise FileNotFoundError(f"{DEATH_REPLAY_DIR / 'death_replay.py'} not found")
        snap = snapshot_buffer(buffer)
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
        self._future = self._pool.submit(_draw, snap, int(update), self.config, self.workers)
        self.update = int(update)
        return True

    def poll(self):
        """None while nothing has finished, else (path or None, summary or error text)."""
        if self._future is None or not self._future.done():
            return None
        fut, self._future = self._future, None
        exc = fut.exception()
        if exc is not None:
            if isinstance(exc, BrokenProcessPool):
                self.close()                                         # a dead worker: start a new one next time
            return None, f"{type(exc).__name__}: {exc}"
        path, summary, _log = fut.result()
        return path, summary

    def close(self, wait=False):
        """Stops the worker once the map being drawn is done (wait=False: training does not wait for it)."""
        if self._pool is not None:
            self._pool.shutdown(wait=wait)
            self._pool = None
