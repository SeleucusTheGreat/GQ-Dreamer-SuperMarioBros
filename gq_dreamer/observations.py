"""Model-side observation processing: RGB frames, sprite masks, RAM records and batching of env observations."""
import numpy as np
import torch
import torch.nn.functional as F

from .game import BLK_SPRITE_MASK, BLOCK_COLS, BLOCK_ROWS, GRID_DIM, PATCH, SCREEN_H, SCREEN_W

# --- Model frame geometry (96x120 = native x 0.75) ----------------------------------------------------------------
IMG_H, IMG_W = SCREEN_H * 3 // 4, SCREEN_W * 3 // 4                  # 96, 120 (exact area downsampling)
TILE_PX = PATCH * 3 // 4                                              # 12: one game tile in the model frame
# square patches of k x k game tiles (k = enc_patch_tiles): the MAE masking grid and the tokens of the "linear" stem;
# k = 1: 80 patches of 12x12 (one per tile), k = 2: 20 patches of 24x24


def patch_grid(k=1, native=False):
    """(rows, columns, pixels per side) of the k x k-tile patch grid of the 96x120 frame (native: 128x160)."""
    assert BLOCK_ROWS % k == 0 and BLOCK_COLS % k == 0, f"{k} x {k}-tile patches do not tile the 8 x 10 screen"
    return BLOCK_ROWS // k, BLOCK_COLS // k, (PATCH if native else TILE_PX) * k


def n_patches(k=1):
    """Number of k x k-tile patches (80 for k = 1, 20 for k = 2), at either resolution."""
    rows, cols, _ = patch_grid(k)
    return rows * cols


def patch_dim(k=1, native=False):
    """Pixel values per k x k-tile patch: 432 / 1728 for k = 1 / 2 at 96x120, 768 / 3072 at 128x160."""
    return 3 * patch_grid(k, native)[2] ** 2

# tokens of the "cnn" encoder stem: an 8x5 grid, one token per 1x2 tiles (12x24 px), overlapping receptive fields
CNN_ROWS, CNN_COLS = BLOCK_ROWS, BLOCK_COLS // 2                      # 8, 5
N_CNN_TOKENS = CNN_ROWS * CNN_COLS                                    # 40

# --- Side inputs ----------------------------------------------------------------------------------------------------
NUM_MAP_IDS = 256
N_TIERS = 3
STATUS_DIM = 6                     # tier one-hot x3, star, lives / 10, facing right
MODE_DIM = 2                       # in gameplay, on the world map
RAM_DIM = 6                        # stored RAM record: tier, star, lives, facing, gameplay, on map
GRID_BYTES = (GRID_DIM + 7) // 8


# --- Frames ---------------------------------------------------------------------------------------------------------
def autocast_dtype(device_type):
    """Current autocast dtype of this device, fp32 when autocast is off."""
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return torch.float32


def screen_rgb(screen, palette, dtype=torch.float32):
    """Palette-indexed screen [N, 128, 160] + palette [N, P, 3] -> RGB [N, 3, 128, 160] in [0, 1]."""
    N = screen.shape[0]
    pal = palette.to(dtype) / 255.0
    idx = screen.reshape(N, -1, 1).long().expand(-1, -1, 3)
    return torch.gather(pal, 1, idx).view(N, SCREEN_H, SCREEN_W, 3).permute(0, 3, 1, 2)


def _area_matrix(n_in, n_out):
    """[n_out, n_in] exact area-averaging weights."""
    s = n_in / n_out
    m = torch.zeros(n_out, n_in, dtype=torch.float64)
    for j in range(n_out):
        a, b = j * s, (j + 1) * s
        for i in range(int(a), min(n_in, int(-(-b // 1)))):
            m[j, i] = max(0.0, min(b, i + 1) - max(a, i))
    return m / s


_AREA = {}


def downsample(rgb):
    """[N, 3, 128, 160] -> [N, 3, 96, 120] by exact area averaging, in rgb's own dtype (autocast off)."""
    key = (rgb.device, rgb.dtype)
    if key not in _AREA:
        _AREA[key] = (_area_matrix(SCREEN_H, IMG_H).to(rgb.device, rgb.dtype),
                      _area_matrix(SCREEN_W, IMG_W).to(rgb.device, rgb.dtype))
    ah, aw = _AREA[key]
    with torch.autocast(rgb.device.type, enabled=False):
        return torch.einsum("hH,ncHW,wW->nchw", ah, rgb, aw)


def frame_rgb(screen, palette, dtype=torch.float32):
    """Palette-indexed screen + palette -> the model's RGB frame [N, 3, 96, 120] in [0, 1] (native x 0.75)."""
    return downsample(screen_rgb(screen, palette, dtype)).clamp(0.0, 1.0)


def patchify(img, k=1):
    """[N, 3, H, W] (96x120 or native 128x160) -> [N, n_patches(k), 3 * px * px]: pixels of each k x k-tile patch,
    row-major."""
    N = img.shape[0]
    rows, cols, _ = patch_grid(k)
    px = img.shape[-1] // cols
    x = img.reshape(N, 3, rows, px, cols, px)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(N, rows * cols, 3 * px * px)


def unpatchify(patches, k=1):
    """[N, n_patches(k), patch_dim(k)] -> [N, 3, 96, 120] (inverse of patchify)."""
    N = patches.shape[0]
    rows, cols, px = patch_grid(k)
    x = patches.reshape(N, rows, cols, 3, px, px)
    return x.permute(0, 3, 1, 4, 2, 5).reshape(N, 3, rows * px, cols * px)


def frame_input(screen, palette, dtype=torch.float32, chunk=1024, native=False):
    """Palette-indexed screens [N, 128, 160] + palettes [N, P, 3] -> encoder input frames [N, 3, 96, 120] in [-1, 1]
    (native: [N, 3, 128, 160], the original screen without downsampling).

    The pixels need no gradient, so the frames are converted `chunk` at a time without one: the int64 palette indices
    and the fp32 RGB only ever exist for `chunk` frames, and each chunk is written straight into the one output tensor
    in `dtype` (the encoder's compute dtype)."""
    N = screen.shape[0]
    h, w = (SCREEN_H, SCREEN_W) if native else (IMG_H, IMG_W)
    out = torch.empty(N, 3, h, w, dtype=dtype, device=screen.device)
    to_rgb = screen_rgb if native else frame_rgb
    with torch.no_grad():
        for i in range(0, N, chunk):
            rgb = to_rgb(screen[i:i + chunk], palette[i:i + chunk]).clamp(0.0, 1.0)   # fp32 [n, 3, h, w] in [0, 1]
            out[i:i + chunk] = rgb.mul_(2.0).sub_(1.0)
    return out


def patch_pixel_mask(hide, k=1):
    """Patch mask [N, n_patches(k)] -> pixel mask [N, 1, 96, 120] (True on the pixels of the hidden patches)."""
    N = hide.shape[0]
    rows, cols, px = patch_grid(k)
    m = hide.view(N, 1, rows, 1, cols, 1).expand(N, 1, rows, px, cols, px)
    return m.reshape(N, 1, IMG_H, IMG_W)


# --- Sprite masks ---------------------------------------------------------------------------------------------------
def sprite_tiles(blocks):
    """Overlay [N, 8, 10] -> bool [N, 8, 10]: tiles that contain a sprite."""
    return (blocks.long() & BLK_SPRITE_MASK) > 0


def sprite_pixel_mask(blocks):
    """Overlay [N, 8, 10] -> bool [N, 96, 120]: pixels of sprite tiles (12 px per tile in the model frame)."""
    return sprite_tiles(blocks).repeat_interleave(TILE_PX, dim=1).repeat_interleave(TILE_PX, dim=2)


def sprite_patches(blocks, k=1):
    """Overlay [N, 8, 10] -> bool [N, n_patches(k)]: k x k-tile patches that contain a sprite."""
    N = blocks.shape[0]
    rows, cols, _ = patch_grid(k)
    t = sprite_tiles(blocks).view(N, rows, k, cols, k)
    return t.any(dim=4).any(dim=2).flatten(1)


def sample_hide(blocks, mask_max=0.9, k=1):
    """MAE mask [N, n_patches(k)]: each frame hides p ~ U(0, mask_max) of its background patches (sprite patches
    never)."""
    N, dev = blocks.shape[0], blocks.device
    p = torch.rand(N, 1, device=dev) * mask_max
    return (torch.rand(N, n_patches(k), device=dev) < p) & ~sprite_patches(blocks, k)


# --- RAM record -----------------------------------------------------------------------------------------------------
def obs_ram(obs):
    """Env observation -> stored RAM record [6] uint8 (tier, star, lives, facing, gameplay, on map)."""
    s = np.asarray(obs["state"], dtype=np.float32)
    return np.array([int(np.argmax(s[0:3])), s[3] > 0.5, int(round(float(s[4]) * 10.0)), s[7] > 0.5,
                     s[12] > 0.5, s[13] > 0.5], dtype=np.uint8)


def ram_fields(ram):
    """RAM record [..., 6] -> dict of float tensors (status, mode, tier, star, facing, lives)."""
    r = ram.long()
    tier = r[..., 0].clamp(0, N_TIERS - 1)
    lives = r[..., 2].clamp(max=20).float() / 10.0
    status = torch.cat([F.one_hot(tier, N_TIERS).float(), r[..., 1:2].float(), lives[..., None], r[..., 3:4].float()],
                       dim=-1)
    return {"status": status, "mode": r[..., 4:6].float(), "tier": tier, "star": r[..., 1].float(),
            "facing": r[..., 3].float(), "lives": lives}


def status_input(obs):
    """Status-token input [N, 6]: RAM status (tier one-hot, star, lives, facing); no level timer."""
    return ram_fields(obs["ram"])["status"]


# --- Bit packing ----------------------------------------------------------------------------------------------------
def pack_bits(v):
    """Binary vector -> little-endian packed uint8 bytes."""
    return np.packbits(np.asarray(v) > 0.5, bitorder="little")


_SHIFTS = {}


def unpack_bits(p, n):
    """Packed bytes [..., B] uint8 -> float bits [..., n]."""
    if p.device not in _SHIFTS:
        _SHIFTS[p.device] = torch.arange(8, dtype=torch.uint8, device=p.device)
    bits = (p.unsqueeze(-1) >> _SHIFTS[p.device]) & 1
    return bits.reshape(*p.shape[:-1], p.shape[-1] * 8)[..., :n].float()


# --- Batching -------------------------------------------------------------------------------------------------------
def obs_to_tensors(obs_list, device):
    """List of env observations -> batched encoder inputs on device."""
    return {
        "screen": torch.from_numpy(np.stack([o["screen"] for o in obs_list])).to(device),
        "palette": torch.from_numpy(np.stack([o["palette"] for o in obs_list])).to(device),
        "grid": torch.from_numpy(np.stack([o["grid"] for o in obs_list])).float().to(device),
        "map_id": torch.tensor([int(o["map_id"]) for o in obs_list], dtype=torch.long, device=device),
        "ram": torch.from_numpy(np.stack([obs_ram(o) for o in obs_list])).to(device),
    }
