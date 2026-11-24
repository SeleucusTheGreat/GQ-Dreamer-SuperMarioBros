"""PyBoy environment for Super Mario Bros. Deluxe (GBC, 'Original 1985' mode) and its parallel wrappers."""
import io
import multiprocessing as mp
import os
import warnings

import gymnasium as gym
import numpy as np
from gymnasium import spaces

warnings.filterwarnings("ignore", message="Using SDL2 binaries")    # pysdl2-dll banner
from pyboy import PyBoy
from pyboy.utils import WindowEvent

from .game import (ACTIONS, ACTION_NAMES, BLK_VALID_BIT, BLOCK_COLS, BLOCK_ROWS, FRAMES_PER_STEP, GRID_DIM,
                   GRID_RADIUS, HUD_CROP_PX, LTM_COMPLETED_OFFSET, LTM_FIRE_INDEX, LTM_MUSHROOM_INDEX,
                   LTM_REACHED_OFFSET, LTM_REWARD_DIM, LTM_STAR_INDEX, MAP_CLASS_INDEX, MAP_CURIOSITY_TABLE, MAP_NAMES,
                   NUM_LEVELS, PALETTE_SIZE, PATCH, SCREEN_H, SCREEN_W, STATE_DIM, TILE_SIZE)

START_STATE = "SMBDeluxe.Start.state"


# --- Palette compression --------------------------------------------------------------------------------------------
def palettize(rgb, size=PALETTE_SIZE):
    """[H, W, 3] uint8 -> (palette index [H, W], palette [size, 3]); extra colours map to the nearest kept one."""
    key = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2].astype(np.int32)
    uniq, inv, counts = np.unique(key.ravel(), return_inverse=True, return_counts=True)
    cols = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=-1).astype(np.int32)
    if len(uniq) > size:
        keep = np.argsort(-counts)[:size]
        d = ((cols[:, None, :] - cols[None, keep, :]) ** 2).sum(-1)
        inv = np.argmin(d, axis=1)[inv]
        cols = cols[keep]
    palette = np.zeros((size, 3), dtype=np.uint8)
    palette[:len(cols)] = cols
    return inv.reshape(rgb.shape[:2]).astype(np.uint8), palette


# per-process lookup tables of palettize_screen
_KEY_POS = np.zeros(1 << 15, dtype=np.intp)
_KEY_LUT = np.zeros(1 << 15, dtype=np.uint8)
_PIX_IDX = np.arange(SCREEN_H * SCREEN_W, dtype=np.intp)


def palettize_screen(screen_rgba, size=PALETTE_SIZE):
    """Fast palettize of PyBoy's RGBA screen (HUD cropped) via 15-bit GBC colour keys; exact fallback on collisions."""
    crop = screen_rgba[HUD_CROP_PX:]
    if crop.shape != (SCREEN_H, SCREEN_W, 4) or not crop.flags["C_CONTIGUOUS"]:
        return palettize(crop[..., :3], size)
    v = crop.view(np.uint32).reshape(-1)                             # packed RGBA, R = low byte
    key = ((((v >> 3) & 0x1F) << 10) | (((v >> 11) & 0x1F) << 5) | ((v >> 19) & 0x1F)).astype(np.intp)
    uniq = np.flatnonzero(np.bincount(key, minlength=1 << 15))
    if len(uniq) > size:
        return palettize(crop[..., :3], size)
    _KEY_POS[key] = _PIX_IDX
    pos = _KEY_POS[uniq]
    _KEY_LUT[uniq] = np.arange(len(uniq), dtype=np.uint8)
    idx = _KEY_LUT[key]
    if not np.array_equal(v[pos][idx], v):                           # two colours shared a 15-bit key
        return palettize(crop[..., :3], size)
    palette = np.zeros((size, 3), dtype=np.uint8)
    palette[:len(uniq)] = crop[..., :3].reshape(-1, 3)[pos]
    return idx.reshape(SCREEN_H, SCREEN_W), palette


# --- Environment ----------------------------------------------------------------------------------------------------
class SuperMarioEnv(gym.Env):
    """Gym env over PyBoy; RAM map from datacrystal.tcrf.net/wiki/Super_Mario_Bros._Deluxe/RAM_map."""

    # WRAM / HRAM addresses
    RAM_LEVEL_ID = 0xC162
    RAM_SCORE = 0xC17A          # 3 BCD bytes
    RAM_TIMER = 0xC17D          # 2 bytes, level timer (binary): 400 a few steps into a level, -1 every 24 frames;
                                # reads 0 in a level's first frames and after the flagpole tally
    RAM_LIVES = 0xC17F
    RAM_PLAYER_STATE = 0xC1C1   # 03 dead, 04 on map
    RAM_DIRECTION = 0xC1C3      # 00 left, 02 right
    RAM_GROWTH = 0xC1C5         # 00 small, 01 big
    RAM_X = 0xC1CA              # 2 bytes, x in level (px)
    RAM_Y = 0xC1CC              # 2 bytes
    RAM_FIREBALLS = 0xC1CE      # 01 fire flower
    RAM_STAR_TIMER = 0xC1DA     # 2 bytes
    RAM_COINS = 0xC1F2
    RAM_MODE = 0xFFB5
    RAM_AREA_TYPE = 0xC36F      # area graphics type: 0x62 overworld, 0x63 underground, 0x68 castle, 0x66 water (found
                                # by RAM diff over all 32 levels: 0x66 only in 2-2 and 7-2)

    STATE_DEAD = 0x03
    STATE_ON_MAP = 0x04
    STATE_PIT = 0x11            # fell into a pit (not in the datacrystal map; seen only before pit deaths)
    AREA_WATER = 0x66
    FIRST_LEVEL_ID = 0x00
    PIT_DEATH_Y = 240           # at or below this y Mario is inside a pit
    PIT_STATE_MIN_Y = 224       # STATE_PIT counts as a death only this low (water pits freeze y at 238 / 239)

    GAMEPLAY_MODES = frozenset((0x06, 0x09, 0x0A, 0x0B, 0x0D))
    MAP_MODES = frozenset((0x04, 0x05, 0x07, 0x08))
    TERMINAL_MODES = frozenset((0x0E, 0x0F, 0x10, 0x11, 0x14, 0x1A, 0x1B, 0x22))
    CREDITS_MODES = frozenset((0x39, 0x3A))

    BTN = {
        "left": (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
        "right": (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
        "up": (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
        "down": (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
        "a": (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
        "b": (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
    }
    ACTIONS = ACTIONS
    ACTION_NAMES = ACTION_NAMES

    SKIP_MAX_STEPS = 1000                                            # cap of one skipped transition (agent steps)
    # Python-side state that step() reads: copied into / restored from an env snapshot (the emulator state aside)
    SNAPSHOT_ATTRS = ("current_step", "max_step_limit", "max_events", "reached_levels", "completed_levels",
                      "got_mushroom", "got_fire", "got_star", "start_event_bits", "last_reward_bits",
                      "best_powerup_tier", "last_coins", "_active_level", "_was_in_gameplay", "_death_latch",
                      "_timed_out", "_max_lives", "episode_visited_tiles", "_last_tile_coord", "_held")

    def __init__(self, rom_path, state_path, verbose=False, window="SDL2", speed=0, frames_per_step=FRAMES_PER_STEP,
                 skip_transitions=False):
        super().__init__()
        self.state_path = state_path
        self.verbose = verbose
        self.frames_per_step = frames_per_step
        self._render_all = window == "SDL2"                          # render every frame when a human watches
        self._capture = None                                         # list while recording: every emulator frame

        try:                                                         # sound_emulated=False skips the APU (~15% faster)
            self.pyboy = PyBoy(rom_path, window=window, sound_volume=0, sound_emulated=False, cgb=True)
        except TypeError:                                            # PyBoy without the sound_emulated option
            self.pyboy = PyBoy(rom_path, window=window, sound_volume=0, cgb=True)
        self.pyboy.set_emulation_speed(speed)

        # step budget in game time: ~400 s of gameplay, +~133 s per completed level
        self.init_steps = 24000 // frames_per_step
        self.step_increase = 8000 // frames_per_step
        self.reward_event_val = 50                                   # level completed
        self.reward_powerup_val = 10.0                               # per power-up tier gained
        self.reward_coin_val = 1.0                                   # per coin

        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Dict({
            "screen": spaces.Box(0, PALETTE_SIZE - 1, (SCREEN_H, SCREEN_W), np.uint8),
            "palette": spaces.Box(0, 255, (PALETTE_SIZE, 3), np.uint8),
            "blocks": spaces.Box(0, 255, (BLOCK_ROWS, BLOCK_COLS), np.uint8),
            "state": spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32),
            "events": spaces.Box(0, 1, (LTM_REWARD_DIM,), np.float32),
            "grid": spaces.Box(0, 1, (GRID_DIM,), np.float32),
        })
        self._held = set()
        self.water_repress = True                                    # under water every A action is a new press
        self.timeout_truncates = True                                # False = a time-out ends the episode as a death
        self.skip_transitions = skip_transitions                     # True: one agent step spans every non-playable
                                                                     # phase (world map, transition screens)

        self.current_step = 0
        self.max_step_limit = self.init_steps
        self.max_events = 0
        self.reached_levels = np.zeros(NUM_LEVELS, dtype=np.float32)
        self.completed_levels = np.zeros(NUM_LEVELS, dtype=np.float32)
        self.got_mushroom = self.got_fire = self.got_star = False
        self.start_event_bits = np.zeros(LTM_REWARD_DIM, dtype=np.float32)
        self.last_reward_bits = np.zeros(LTM_REWARD_DIM, dtype=np.float32)
        self.best_powerup_tier = 0
        self.last_coins = 0
        self._active_level = None
        self._was_in_gameplay = False
        self._death_latch = False
        self._timed_out = False                                      # the latch fired with the level timer at 0
        self._max_lives = 0
        self.episode_visited_tiles = set()
        self._last_tile_coord = (0, 0, 0)

    # --- RAM helpers ------------------------------------------------------------------------------------------------
    def _read16(self, addr):
        """Little-endian 16-bit RAM read."""
        return self.pyboy.memory[addr] | (self.pyboy.memory[addr + 1] << 8)

    def _mode(self):
        """Game mode byte (gameplay, world map, transitions...)."""
        return self.pyboy.memory[self.RAM_MODE]

    def _player_state(self):
        """Player state byte (dead, on map...)."""
        return self.pyboy.memory[self.RAM_PLAYER_STATE]

    def _level_id(self):
        """Current map id."""
        return self.pyboy.memory[self.RAM_LEVEL_ID]

    def _in_gameplay(self, mode=None):
        """True while a level is being played."""
        return (mode if mode is not None else self._mode()) in self.GAMEPLAY_MODES

    def _powerup_tier(self):
        """0 = small, 1 = big, 2 = fire."""
        if self.pyboy.memory[self.RAM_FIREBALLS] == 1:
            return 2
        return 1 if self.pyboy.memory[self.RAM_GROWTH] >= 1 else 0

    def _level_timer(self):
        """Level timer (400 -> 0); 0 also in a level's first frames and after the flagpole tally."""
        return self._read16(self.RAM_TIMER)

    def _star_active(self):
        """True while the star power-up is running."""
        return self._read16(self.RAM_STAR_TIMER) > 0

    def _get_score(self):
        """Score decoded from its 3 BCD bytes."""
        score = 0
        for i in range(3):
            b = self.pyboy.memory[self.RAM_SCORE + i]
            score += ((b >> 4) * 10 + (b & 0x0F)) * (100 ** i)
        return score

    def _get_current_position(self):
        """(map id, x tile, y tile) in 16 px level tiles."""
        return self._level_id(), self._read16(self.RAM_X) // TILE_SIZE, self._read16(self.RAM_Y) // TILE_SIZE

    # --- Observation ------------------------------------------------------------------------------------------------
    def _get_explored_grid(self, coord):
        """5x5 explored grid around Mario, read before the current tile is marked."""
        map_id, x, y = coord
        grid = np.zeros(GRID_DIM, dtype=np.float32)
        i = 0
        for dy in range(-GRID_RADIUS, GRID_RADIUS + 1):
            for dx in range(-GRID_RADIUS, GRID_RADIUS + 1):
                grid[i] = 1.0 if (map_id, x + dx, y + dy) in self.episode_visited_tiles else 0.0
                i += 1
        return grid

    def _get_blocks(self):
        """[8, 10] overlay of the screen tiles: OAM sprite coverage bin + gameplay-valid bit."""
        cover = np.zeros((SCREEN_H, SCREEN_W), dtype=np.uint8)
        for i in range(40):
            s = self.pyboy.get_sprite(i)
            if not s.on_screen:
                continue
            w, h = s.shape
            x0, y0 = max(s.x, 0), max(s.y - HUD_CROP_PX, 0)
            x1, y1 = min(s.x + w, SCREEN_W), min(s.y + h - HUD_CROP_PX, SCREEN_H)
            if x1 > x0 and y1 > y0:
                cover[y0:y1, x0:x1] = 1
        area = cover.reshape(BLOCK_ROWS, PATCH, BLOCK_COLS, PATCH).sum(axis=(1, 3))
        bins = np.digitize(area, (1, 64, 128)).astype(np.uint8)        # none / <25% / <50% / >=50%
        if self._in_gameplay():
            bins |= np.uint8(1 << BLK_VALID_BIT)
        return bins

    def _get_state_vector(self, coord):
        """Compact RAM state vector [STATE_DIM]."""
        s = np.zeros(STATE_DIM, dtype=np.float32)
        s[self._powerup_tier()] = 1.0
        s[3] = 1.0 if self._star_active() else 0.0
        s[4] = min(self.pyboy.memory[self.RAM_LIVES], 20) / 10.0
        lid = coord[0]
        s[5] = (lid // 4 + 1) / 8.0 if lid < NUM_LEVELS else 0.0
        s[6] = (lid % 4 + 1) / 4.0 if lid < NUM_LEVELS else 0.0
        s[7] = 1.0 if self.pyboy.memory[self.RAM_DIRECTION] == 2 else 0.0
        s[8] = self.pyboy.memory[self.RAM_COINS] / 100.0
        s[9] = np.log1p(self._get_score()) / 12.0
        s[10] = min(self._read16(self.RAM_X), 8191) / 4096.0
        s[11] = min(self._read16(self.RAM_Y), 511) / 256.0
        mode = self._mode()
        s[12] = 1.0 if mode in self.GAMEPLAY_MODES else 0.0
        s[13] = 1.0 if mode in self.MAP_MODES else 0.0
        s[14] = np.log1p(max(self.max_step_limit - self.current_step, 0)) / 9.0
        return s

    def _observe(self, coord, grid):
        """Observation dict of the current frame."""
        idx, palette = palettize_screen(self.pyboy.screen.ndarray)
        return {
            "screen": idx,
            "palette": palette,
            "blocks": self._get_blocks(),
            "state": self._get_state_vector(coord),
            "events": self._get_ltm_reward(),
            "grid": grid,
            "map_id": int(coord[0]),
        }

    # --- Input ------------------------------------------------------------------------------------------------------
    def _release_all(self):
        """Releases every held button."""
        for name in self._held:
            self.pyboy.send_input(self.BTN[name][1])
        self._held.clear()

    def _tick(self, count, render_last=True):
        """Advances the emulator, rendering only the last frame unless a human watches or frames are captured."""
        if count <= 0:
            return
        if self._capture is not None:
            for _ in range(count):
                self.pyboy.tick(1, True)
                self._capture.append(self.rgb_frame())
            return
        if self._render_all:
            for _ in range(count):
                self.pyboy.tick(1, True)
        else:
            if count > 1:
                self.pyboy.tick(count - 1, False)
            self.pyboy.tick(1, render_last)

    def _in_water(self):
        """True in underwater areas (2-2, 7-2, ...), where every swim stroke needs a new A press."""
        return self.pyboy.memory[self.RAM_AREA_TYPE] == self.AREA_WATER

    def _apply_action(self, action):
        """Holds the action's buttons for one step (sticky buttons); taps A on the world map. Under water, an A that
        was already held is released for one frame and pressed again, so every A action is a swim stroke."""
        if self._mode() in self.MAP_MODES:                           # the map needs a fresh A press edge
            self._release_all()
            self._tick(2, render_last=False)
            self.pyboy.send_input(self.BTN["a"][0])
            self._tick(max(self.frames_per_step - 3, 2), render_last=False)
            self.pyboy.send_input(self.BTN["a"][1])
            self._tick(1)
            return
        desired = set(self.ACTIONS[action])
        repress = self.water_repress and "a" in desired and "a" in self._held and self._in_water()
        for name in self._held - desired:
            self.pyboy.send_input(self.BTN[name][1])
        for name in desired - self._held:
            self.pyboy.send_input(self.BTN[name][0])
        self._held = desired
        if repress:                                                  # held A gives no stroke: make a new press edge
            self.pyboy.send_input(self.BTN["a"][1])
            self._tick(1, render_last=False)
            self.pyboy.send_input(self.BTN["a"][0])
            self._tick(self.frames_per_step - 1)
        else:
            self._tick(self.frames_per_step)

    # --- Progress / death -------------------------------------------------------------------------------------------
    def _update_death_latch(self):
        """Sticky death detection: dead state, fell into a pit, or a life lost. A death whose latch fires with the
        level timer at 0 is a time-out (the game kills Mario 3 steps after the timer reaches 0)."""
        was_latched = self._death_latch
        state = self._player_state()
        lives = self.pyboy.memory[self.RAM_LIVES]
        y = self._read16(self.RAM_Y)
        if state == self.STATE_DEAD:
            self._death_latch = True
        # pit: the game sets STATE_PIT when Mario falls into one; in a water pit (2-2 seabed gap) y then freezes at
        # 238 / 239 and the life is only lost 26-27 steps later, so y >= 240 alone misses most water-pit deaths
        if self._in_gameplay() and state == self.STATE_PIT and y >= self.PIT_STATE_MIN_Y:
            self._death_latch = True
        if self._in_gameplay() and state != self.STATE_ON_MAP and y >= self.PIT_DEATH_Y:
            self._death_latch = True
        if lives < self._max_lives:
            self._death_latch = True
        self._max_lives = max(self._max_lives, lives)
        if self._death_latch and not was_latched:
            self._timed_out = self._in_gameplay() and self._level_timer() == 0

    def _update_progress_flags(self):
        """Updates reached / completed levels and power-ups; returns log lines."""
        self._update_death_latch()
        mode, state, lid = self._mode(), self._player_state(), self._level_id()
        logs = []
        in_gameplay = self._in_gameplay(mode)
        if in_gameplay and not self._death_latch and state != self.STATE_ON_MAP:
            if lid < NUM_LEVELS:
                self.reached_levels[lid] = 1.0
                self._active_level = lid
            tier = self._powerup_tier()
            self.got_mushroom |= tier >= 1
            self.got_fire |= tier >= 2
            self.got_star |= self._star_active()
        # alive gameplay -> world map transition = flagpole / axe reached
        if (mode in self.MAP_MODES or mode in self.CREDITS_MODES) and self._was_in_gameplay \
                and not self._death_latch and self._active_level is not None:
            if self.completed_levels[self._active_level] == 0.0:
                self.completed_levels[self._active_level] = 1.0
                name = MAP_NAMES.get(self._active_level, self._active_level)
                logs.append(f"\033[1;92m>>> [LEVEL COMPLETE] {name} <<<\033[0m")
        self._was_in_gameplay = in_gameplay
        return logs

    def _get_event_bits(self):
        """Progress bits of the episode so far [LTM_REWARD_DIM]."""
        bits = np.zeros(LTM_REWARD_DIM, dtype=np.float32)
        bits[LTM_REACHED_OFFSET:LTM_REACHED_OFFSET + NUM_LEVELS] = self.reached_levels
        bits[LTM_COMPLETED_OFFSET:LTM_COMPLETED_OFFSET + NUM_LEVELS] = self.completed_levels
        bits[LTM_MUSHROOM_INDEX] = float(self.got_mushroom)
        bits[LTM_FIRE_INDEX] = float(self.got_fire)
        bits[LTM_STAR_INDEX] = float(self.got_star)
        return bits

    def _get_ltm_reward(self):
        """Progress bits gained since the episode start."""
        return self._get_event_bits() * (1.0 - self.start_event_bits)

    def _get_reward_event_bits(self):
        """New progress bits that pay the sparse reward (completed levels only)."""
        out = np.zeros(LTM_REWARD_DIM, dtype=np.float32)
        new = self._get_ltm_reward()
        out[LTM_COMPLETED_OFFSET:LTM_MUSHROOM_INDEX] = new[LTM_COMPLETED_OFFSET:LTM_MUSHROOM_INDEX]
        return out

    @staticmethod
    def _describe_ltm_bit(idx):
        """Readable name of a progress bit."""
        if idx < LTM_COMPLETED_OFFSET:
            return f"reached {MAP_NAMES.get(idx, idx)}"
        if idx < LTM_MUSHROOM_INDEX:
            return f"completed {MAP_NAMES.get(idx - LTM_COMPLETED_OFFSET, idx)}"
        return {LTM_MUSHROOM_INDEX: "first Mushroom", LTM_FIRE_INDEX: "first Fire Flower",
                LTM_STAR_INDEX: "first Star"}[idx]

    # --- Gym API ----------------------------------------------------------------------------------------------------
    def step(self, action):
        """One agent step: act, then compute curiosity, rewards and termination."""
        self._apply_action(action)
        self.current_step += 1
        sparse_reward, standard_reward = 0.0, 0.0
        logs = self._update_progress_flags()
        skipped = self._skip_transition(logs) if self.skip_transitions else 0

        mode, state = self._mode(), self._player_state()
        coord = self._get_current_position()
        map_id, x, _ = coord

        # tile curiosity: a new tile pays its map category (x2 when entered moving right)
        grid = self._get_explored_grid(coord)
        curiosity, tier_event = 0.0, 0.0
        if self._in_gameplay(mode) and state not in (self.STATE_DEAD, self.STATE_ON_MAP):
            if coord not in self.episode_visited_tiles:
                self.episode_visited_tiles.add(coord)
                curiosity = float(MAP_CURIOSITY_TABLE[map_id])
                last_map, last_x, _ = self._last_tile_coord
                if map_id == last_map and x > last_x:
                    curiosity *= 2.0
                if MAP_CLASS_INDEX[map_id] > 0:
                    tier_event = 1.0
            self._last_tile_coord = coord

        # level completion (+50) and budget extension
        reward_bits = self._get_reward_event_bits()
        n_events = int(reward_bits.sum())
        if n_events > self.max_events:
            new_events = n_events - self.max_events
            for idx in np.where((reward_bits > 0.5) & (self.last_reward_bits < 0.5))[0]:
                logs.append(f"\033[1;95m[EVENT REWARD] +{self.reward_event_val:.1f} | "
                            f"{self._describe_ltm_bit(int(idx))}\033[0m")
            self.max_step_limit += new_events * self.step_increase
            self.max_events = n_events
            sparse_reward += new_events * self.reward_event_val
        self.last_reward_bits = reward_bits

        # power-ups pay only when the episode's best tier rises (the growth animation flickers the RAM flag)
        tier = self._powerup_tier()
        if tier > self.best_powerup_tier:
            gain = (tier - self.best_powerup_tier) * self.reward_powerup_val
            standard_reward += gain
            if self.verbose:
                logs.append(f"\033[1;92m[POWERUP] +{gain:.1f} | tier {self.best_powerup_tier} -> {tier}\033[0m")
            self.best_powerup_tier = tier
        coins = self.pyboy.memory[self.RAM_COINS]
        coin_delta = coins - self.last_coins
        if coin_delta < 0:
            coin_delta += 100                                        # 100-coin wrap (1-up)
        if 0 < coin_delta <= 20:
            standard_reward += coin_delta * self.reward_coin_val
        self.last_coins = coins

        # termination (death / game over / game beaten) vs truncation (step budget, level time-out). The model does
        # not see the level timer (it sits in the cropped HUD), so a time-out is a truncation, not a death.
        game_beaten = mode in self.CREDITS_MODES
        timed_out = self._death_latch and self._timed_out and self.timeout_truncates
        died = self._death_latch and not timed_out
        terminated = not timed_out and (self._death_latch or (mode in self.TERMINAL_MODES) or game_beaten)
        truncated = timed_out or self.current_step >= self.max_step_limit
        if terminated and self.verbose:
            reason = "GAME BEATEN!" if game_beaten else "died" if died else f"terminal mode {hex(mode)}"
            logs.append(f"\033[1;91m[TERMINATED] {reason} at step {self.current_step}\033[0m")
        if timed_out and self.verbose:
            logs.append(f"\033[1;93m[TRUNCATED] time-out at step {self.current_step}\033[0m")

        info = {
            "coord": coord, "curiosity": curiosity, "tier_event": tier_event,
            "sparse_reward": sparse_reward, "standard_reward": standard_reward,
            "steps": self.current_step, "limit": self.max_step_limit, "events": n_events,
            "map_id": int(map_id), "died": bool(died), "timeout": bool(timed_out), "logs": logs,
            "playable": bool(self._in_gameplay(mode)), "mode": int(mode), "skipped": skipped,
        }
        if self._capture is not None:                                # the step's emulator frames, for videos
            info["frames"] = np.stack(self._capture)
            self._capture = []
        return self._observe(coord, grid), sparse_reward + standard_reward, terminated, truncated, info

    def _playable(self, mode=None):
        """True when the agent's action matters: a level is being played (False on the world map and on the
        transition screens between levels)."""
        return self._in_gameplay(mode)

    def _skip_transition(self, logs):
        """skip_transitions: ticks on (tapping A on the world map, as _apply_action does) while the game is in a
        non-playable phase, until gameplay resumes, a death / terminal mode / the credits, or SKIP_MAX_STEPS ->
        number of agent steps skipped. Progress flags are updated every step, so a level completion found on the way
        still pays its reward in this step; the skipped steps do not count against the step budget."""
        n = 0
        while n < self.SKIP_MAX_STEPS and not self._death_latch:
            mode = self._mode()
            if self._playable(mode) or mode in self.TERMINAL_MODES or mode in self.CREDITS_MODES:
                break
            self._apply_action(0)                                    # NOOP (on the map: an A tap)
            logs.extend(self._update_progress_flags())
            n += 1
        return n

    # --- Snapshots (dream audit) ------------------------------------------------------------------------------------
    def get_snapshot(self):
        """Complete state of the env between two steps: the emulator (PyBoy save state) and every Python-side
        internal that step() reads (copies). restore_snapshot + the same actions give the same observations,
        rewards and infos."""
        buf = io.BytesIO()
        self.pyboy.save_state(buf)
        py = {}
        for name in self.SNAPSHOT_ATTRS:
            v = getattr(self, name)
            py[name] = v.copy() if hasattr(v, "copy") else v
        return {"emulator": buf.getvalue(), "py": py}

    def restore_snapshot(self, snap):
        """Restores a get_snapshot state (from any env on the same ROM). The joypad is set to the held buttons again
        (released all, then pressed), so the next action's press / release edges are those of the original run."""
        self.pyboy.load_state(io.BytesIO(snap["emulator"]))
        for name, v in snap["py"].items():
            setattr(self, name, v.copy() if hasattr(v, "copy") else v)
        for name, (_, release) in self.BTN.items():
            self.pyboy.send_input(release)
        for name in self._held:
            self.pyboy.send_input(self.BTN[name][0])
        self._capture = [] if self._capture is not None else None

    def reset(self, seed=None, options=None):
        """Loads the start state and clears the episode's progress."""
        super().reset(seed=seed)
        with open(self.state_path, "rb") as f:
            self.pyboy.load_state(f)
        self._release_all()
        self.current_step = 0
        self.max_step_limit = self.init_steps
        self.episode_visited_tiles = set()
        self.pyboy.tick()

        self.reached_levels[:] = 0.0
        self.completed_levels[:] = 0.0
        self.got_mushroom = self.got_fire = self.got_star = False
        self._active_level = None
        self._was_in_gameplay = self._in_gameplay()
        self._death_latch = False
        self._timed_out = False
        self._max_lives = self.pyboy.memory[self.RAM_LIVES]
        self._update_progress_flags()
        self.reached_levels[self.FIRST_LEVEL_ID] = 1.0              # entering 1-1 is not progress

        start = self._get_current_position()
        self.episode_visited_tiles.add(start)                        # the spawn tile is not a discovery
        self._last_tile_coord = start
        self.start_event_bits = self._get_event_bits()
        self.last_reward_bits = self._get_reward_event_bits()
        self.max_events = int(self.last_reward_bits.sum())
        self.best_powerup_tier = self._powerup_tier()
        self.last_coins = self.pyboy.memory[self.RAM_COINS]

        info = {"coord": start, "curiosity": 0.0, "tier_event": 0.0, "events": self.max_events,
                "map_id": int(start[0]), "steps": 0, "limit": self.max_step_limit, "died": False, "timeout": False,
                "logs": []}
        if self._capture is not None:
            info["frames"] = self.rgb_frame()[None]
            self._capture = []
        return self._observe(start, self._get_explored_grid(start)), info

    def set_capture(self, on):
        """on: every emulator frame is rendered and returned as info["frames"] [n, 144, 160, 3] by step / reset."""
        self._capture = [] if on else None

    def rgb_frame(self):
        """Full 144x160 RGB frame (HUD included)."""
        return np.ascontiguousarray(self.pyboy.screen.ndarray[:, :, :3])

    def close(self):
        """Stops the emulator."""
        self.pyboy.stop()


# --- Vectorised environments ----------------------------------------------------------------------------------------
def _step_env(env, action, snapshot=False):
    """Steps env; with snapshot the info also carries env.get_snapshot() of the state after the step."""
    out = env.step(action)
    if snapshot:
        out[4]["snapshot"] = env.get_snapshot()
    return out


def _env_worker(conn, rom_path, state_path, window, speed, verbose, frames_per_step, skip_transitions=False):
    """Subprocess loop serving one env over a pipe."""
    try:
        env = SuperMarioEnv(rom_path=rom_path, state_path=state_path, window=window, speed=speed, verbose=verbose,
                            frames_per_step=frames_per_step, skip_transitions=skip_transitions)
    except Exception as e:
        try:
            conn.send(("__error__", repr(e)))
        finally:
            conn.close()
        return
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                action, snap = data if isinstance(data, tuple) else (data, False)
                conn.send(_step_env(env, action, snap))
            elif cmd == "reset":
                conn.send(env.reset())
            elif cmd == "restore":                                   # dream audit: continue from a snapshot
                env.restore_snapshot(data)
                conn.send(True)
            elif cmd == "rgb_frame":
                conn.send(env.rgb_frame())
            elif cmd == "capture":
                env.set_capture(data)
                conn.send(True)
            elif cmd == "close":
                break
            else:
                conn.send(("__error__", f"unknown command {cmd!r}"))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass
        conn.close()


class ParallelMarioEnvs:
    """One PyBoy per process; step() sends every command before reading any reply."""

    def __init__(self, num_envs, rom_path, state_path, window="null", speed=0, verbose=False,
                 frames_per_step=FRAMES_PER_STEP, skip_transitions=False):
        ctx = mp.get_context("spawn")                                # PyBoy is not fork-safe
        self.num_envs = num_envs
        self.frames_per_step = frames_per_step
        self.action_space = spaces.Discrete(len(ACTIONS))
        self._parents, self._procs = [], []
        for _ in range(num_envs):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_env_worker, daemon=True,
                               args=(child, rom_path, state_path, window, speed, verbose, frames_per_step,
                                     skip_transitions))
            proc.start()
            child.close()
            self._parents.append(parent)
            self._procs.append(proc)

    def __len__(self):
        return self.num_envs

    @staticmethod
    def _check(result):
        """Raises a worker's error in the main process."""
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], str) and result[0] == "__error__":
            raise RuntimeError(f"Env worker error: {result[1]}")
        return result

    def reset(self):
        """Resets every env -> [(obs, info)]."""
        for p in self._parents:
            p.send(("reset", None))
        return [self._check(p.recv()) for p in self._parents]

    def reset_one(self, i):
        """Resets env i -> (obs, info)."""
        self._parents[i].send(("reset", None))
        return self._check(self._parents[i].recv())

    def rgb_frame(self, i):
        """Full RGB frame of env i."""
        self._parents[i].send(("rgb_frame", None))
        return self._check(self._parents[i].recv())

    def restore(self, snapshots):
        """Env i continues from snapshots[i] (get_snapshot of any env on this ROM; None = left as it is)."""
        sent = [i for i, s in enumerate(snapshots) if s is not None]
        for i in sent:
            self._parents[i].send(("restore", snapshots[i]))
        for i in sent:
            self._check(self._parents[i].recv())

    def set_capture(self, on):
        """Every env returns all its emulator frames as info["frames"] (on) or none (off)."""
        for p in self._parents:
            p.send(("capture", bool(on)))
        for p in self._parents:
            self._check(p.recv())

    def step(self, action_idxs, active=None, snapshot=None):
        """Steps the active envs -> list of step results (None for inactive envs); envs flagged in `snapshot` also put
        their post-step snapshot into the info (taken in the worker)."""
        active = active or [True] * self.num_envs
        snapshot = snapshot or [False] * self.num_envs
        for i, p in enumerate(self._parents):
            if active[i]:
                p.send(("step", (int(action_idxs[i]), bool(snapshot[i]))))
        results = [None] * self.num_envs
        for i, p in enumerate(self._parents):
            if active[i]:
                results[i] = self._check(p.recv())
        return results

    def close(self):
        """Stops every worker."""
        for p in self._parents:
            try:
                p.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


class SequentialMarioEnvs:
    """Same interface as ParallelMarioEnvs, in-process (debugging)."""

    def __init__(self, envs):
        self._envs = envs
        self.num_envs = len(envs)
        self.frames_per_step = envs[0].frames_per_step
        self.action_space = envs[0].action_space

    def __len__(self):
        return self.num_envs

    def reset(self):
        """Resets every env -> [(obs, info)]."""
        return [e.reset() for e in self._envs]

    def reset_one(self, i):
        """Resets env i -> (obs, info)."""
        return self._envs[i].reset()

    def rgb_frame(self, i):
        """Full RGB frame of env i."""
        return self._envs[i].rgb_frame()

    def restore(self, snapshots):
        """Env i continues from snapshots[i] (None = left as it is)."""
        for e, s in zip(self._envs, snapshots):
            if s is not None:
                e.restore_snapshot(s)

    def set_capture(self, on):
        """Every env returns all its emulator frames as info["frames"] (on) or none (off)."""
        for e in self._envs:
            e.set_capture(on)

    def step(self, action_idxs, active=None, snapshot=None):
        """Steps the active envs -> list of step results (None for inactive envs); `snapshot` as ParallelMarioEnvs."""
        active = active or [True] * self.num_envs
        snapshot = snapshot or [False] * self.num_envs
        return [_step_env(e, int(action_idxs[i]), snapshot[i]) if active[i] else None
                for i, e in enumerate(self._envs)]

    def close(self):
        """Closes every env."""
        for e in self._envs:
            e.close()


def create_envs(num_envs, rom_path, state_dir="game", wind="null", speed=0, parallel=True,
                frames_per_step=FRAMES_PER_STEP, skip_transitions=False):
    """Creates `num_envs` environments starting from SMBDeluxe.Start.state."""
    state_path = os.path.join(state_dir, START_STATE)
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Required start state not found: {state_path}. Only '{START_STATE}' is supported.")
    print(f"\033[90m    [setup] launching {num_envs} emulators ({START_STATE}, parallel={parallel}, "
          f"{frames_per_step} frames/step{', transitions skipped' if skip_transitions else ''})\033[0m")
    if parallel:
        envs = ParallelMarioEnvs(num_envs=num_envs, rom_path=rom_path, state_path=state_path, window=wind,
                                 speed=speed, verbose=True, frames_per_step=frames_per_step,
                                 skip_transitions=skip_transitions)
    else:
        envs = SequentialMarioEnvs([SuperMarioEnv(rom_path=rom_path, state_path=state_path, verbose=True,
                                                  window=wind, speed=speed, frames_per_step=frames_per_step,
                                                  skip_transitions=skip_transitions)
                                    for _ in range(num_envs)])
    envs.rom_path, envs.state_path, envs.skip_transitions = rom_path, state_path, skip_transitions
    return envs
