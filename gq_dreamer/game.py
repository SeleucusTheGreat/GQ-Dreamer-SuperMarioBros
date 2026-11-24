"""Super Mario Bros. Deluxe constants: screen geometry, actions, levels, curiosity tables and the level timer."""
import numpy as np

# --- Screen geometry ------------------------------------------------------------------------------------------------
HUD_CROP_PX = 16                                               # top HUD rows removed (score / coins / timer)
SCREEN_H, SCREEN_W = 144 - HUD_CROP_PX, 160                    # 128 x 160, native resolution
PATCH = 16                                                     # one game tile
BLOCK_ROWS, BLOCK_COLS = SCREEN_H // PATCH, SCREEN_W // PATCH  # 8 x 10 tiles
NUM_BLOCKS = BLOCK_ROWS * BLOCK_COLS                           # 80
PALETTE_SIZE = 32                                              # colours stored per frame

# --- Control rate ---------------------------------------------------------------------------------------------------
FRAMES_PER_STEP = 8                                            # emulator frames per agent decision (~133 ms)

# --- Sprite overlay ("blocks") bit layout ---------------------------------------------------------------------------
BLK_SPRITE_MASK = 0x03                                         # bits 0-1: sprite coverage bin of the tile
BLK_VALID_BIT = 2                                              # bit 2: in-level gameplay

# --- Actions --------------------------------------------------------------------------------------------------------
ACTIONS = (
    frozenset(),                      # 0:  NOOP
    frozenset(("right",)),            # 1:  RIGHT
    frozenset(("right", "a")),        # 2:  RIGHT + jump
    frozenset(("right", "b")),        # 3:  RIGHT + run
    frozenset(("right", "a", "b")),   # 4:  RIGHT + run + jump
    frozenset(("a",)),                # 5:  jump
    frozenset(("left",)),             # 6:  LEFT
    frozenset(("left", "a")),         # 7:  LEFT + jump
    frozenset(("left", "b")),         # 8:  LEFT + run
    frozenset(("left", "a", "b")),    # 9:  LEFT + run + jump
    frozenset(("down",)),             # 10: DOWN
    frozenset(("up",)),               # 11: UP
)
ACTION_NAMES = ("NOOP", "RIGHT", "R+A", "R+B", "R+A+B", "A", "LEFT", "L+A", "L+B", "L+A+B", "DOWN", "UP")
ACTION_RIGHT, ACTION_LEFT = 1, 6

# --- Levels and maps ------------------------------------------------------------------------------------------------
NUM_LEVELS = 32                                                # main levels 1-1 .. 8-4 (map ids 0..31)
NUM_MAPS = 256
MAIN_LEVELS = tuple(f"{lid // 4 + 1}-{lid % 4 + 1}" for lid in range(NUM_LEVELS))

MAP_NAMES = {lid: f"World {lid // 4 + 1}-{lid % 4 + 1}" for lid in range(NUM_LEVELS)}
MAP_NAMES.update({
    0x21: "1-1 Bonus", 0x22: "1-2 Bonus", 0x23: "3-1 Bonus", 0x24: "4-1 Bonus",
    0x25: "4-2 Bonus", 0x26: "Cloud Bonus 1", 0x27: "Cloud Bonus 2",
    0x28: "2-1 Bonus", 0x29: "5-1 Bonus", 0x2A: "Cloud Bonus 3",
    0x2B: "6-2 Bonus 1", 0x2C: "6-2 Bonus 2", 0x2D: "Cloud Bonus 4",
    0x2E: "7-1 Bonus", 0x2F: "8-2 Bonus", 0x30: "4-2 Warp Zone",
    0x31: "5-2 Bonus", 0x32: "6-2 Bonus 3", 0x33: "8-1 Bonus",
    0x34: "Pipe Entrance Room", 0x35: "Level Exit Room",
    0x36: "8-4 Repeat 1", 0x37: "8-4 Repeat 2",
    0x38: "8-4 Underwater", 0x39: "8-4 Final",
})
CASTLE_MAP_IDS = frozenset(lid for lid in range(NUM_LEVELS) if lid % 4 == 3) | frozenset((0x36, 0x37, 0x38, 0x39))

# --- Long-term progress bits (events) -------------------------------------------------------------------------------
LTM_REACHED_OFFSET = 0                                         # bits 0..31: reached level i
LTM_COMPLETED_OFFSET = NUM_LEVELS                              # bits 32..63: completed level i
LTM_MUSHROOM_INDEX = 2 * NUM_LEVELS
LTM_FIRE_INDEX = 2 * NUM_LEVELS + 1
LTM_STAR_INDEX = 2 * NUM_LEVELS + 2
LTM_REWARD_DIM = 2 * NUM_LEVELS + 3                            # 67

# --- Tile curiosity per map category --------------------------------------------------------------------------------
CURIOSITY_VALUES = (0.05, 0.20, 0.50)                          # generic room, main level, castle


def map_class_index(map_id):
    """Map category: 2 = castle, 1 = main level, 0 = bonus / pipe room."""
    if map_id in CASTLE_MAP_IDS:
        return 2
    return 1 if map_id < NUM_LEVELS else 0


MAP_CLASS_INDEX = np.array([map_class_index(m) for m in range(NUM_MAPS)], dtype=np.int64)
MAP_CURIOSITY_TABLE = np.array([CURIOSITY_VALUES[c] for c in MAP_CLASS_INDEX], dtype=np.float32)

# --- Local explored grid (5x5 level tiles centred on Mario) ---------------------------------------------------------
GRID_RADIUS = 2
GRID_SIDE = 2 * GRID_RADIUS + 1
GRID_DIM = GRID_SIDE * GRID_SIDE                               # 25
GRID_CENTER_INDEX = GRID_DIM // 2
TILE_SIZE = 16

# --- Env state vector -----------------------------------------------------------------------------------------------
STATE_DIM = 15   # tier x3, star, lives, world, stage, facing, coins, score, x, y, gameplay, world map, budget

# --- Level timer (it sits in the cropped HUD, so the model is given the time left) ----------------------------------
LEVEL_TIMER_STEPS = 1206                                       # steps from a level's first frame to its time-out


def level_steps_of(map_ids):
    """Map ids of one episode -> steps since the current main level started (bonus rooms keep the level's clock)."""
    out = np.zeros(len(map_ids), dtype=np.int32)
    cur, start = -1, 0
    for k, m in enumerate(map_ids):
        m = int(m)
        if m < NUM_LEVELS and m != cur:
            cur, start = m, k
        out[k] = k - start
    return out


def time_left(level_steps):
    """Steps since the level started -> time left on the level timer, 1 (start) .. 0 (time-out)."""
    import torch                                                       # local: env workers import game.py
    return (1.0 - torch.as_tensor(level_steps).float() / LEVEL_TIMER_STEPS).clamp(0.0, 1.0)
