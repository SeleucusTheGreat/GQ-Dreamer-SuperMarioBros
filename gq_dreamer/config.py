"""GQ-Dreamer hyperparameters: one dataclass whose defaults are the GQ-Dreamer configuration."""
from dataclasses import asdict, dataclass, fields, replace

from .game import FRAMES_PER_STEP


@dataclass(frozen=True)
class GQDreamerConfig:
    """All model, loss, replay, behaviour and run-control settings of GQ-Dreamer."""
    config_name: str = "gq"                      # tags checkpoint / metric file names
    action_dim: int = 12
    frames_per_step: int = FRAMES_PER_STEP       # control rate (a checkpoint of another rate is never resumed)
    skip_transitions: bool = False               # env: one agent step spans the world map between levels (change 8;
                                                 # ~12 map steps per level completion)

    # --- Tokenizer: fusion encoder -> z = n_latents one-hot categoricals x classes ----------------------------------
    n_latents: int = 32                          # 32 latent queries -> 32 categoricals
    classes: int = 32
    enc_stem: str = "linear"                     # image tokens: "linear" (patches) or "cnn" (8x5 = 40, small CNN)
    enc_patch_tiles: int = 2                     # linear stem: patches of k x k tiles (2: 20 patches, 1: 80 patches)
    enc_input: str = "model"                     # encoder input: "native" 128x160 screen or "model" 96x120 frame
    enc_dim: int = 256
    enc_depth: int = 6                           # encoder layers
    enc_heads: int = 8
    enc_pos_inject: bool = True                  # 2D sin-cos position of the image tokens re-added in every layer
    enc_swiglu: bool = True                      # encoder feed-forward: SwiGLU as in the TSSM (False: GELU MLP 4 x dim)
    enc_ffn: int = 682                           # SwiGLU hidden: 8/3 x enc_dim, the parameters of the 4x GELU MLP
    enc_grad_checkpoint: bool = True             # recompute encoder activations in the backward pass
    enc_last_queries: bool = True                # last encoder layer computes only the latent queries (same result)
    unimix: float = 0.01
    decoder: str = "conv"                        # pixel decoder: "transformer" (patch queries) or "conv"
    dec_dim: int = 256                           # transformer decoder width / layers / heads
    dec_depth: int = 4
    dec_heads: int = 8
    dec_patch_tiles: int = 2                     # transformer decoder: one query per k x k tiles (2: 20 queries)
    dec_chans: tuple = (128, 128, 96, 64, 32)    # conv decoder channels
    head_hidden: int = 256                       # grounding heads (level / status) on z only

    # --- Reconstruction branch --------------------------------------------------------------------------------------
    recon_frames: int = 2048                      # frames decoded per update (0 = every valid frame)
    mask_frames: int = 1024                       # frames re-encoded with MAE masking + token dropout (decoded)
    mask_weight: float = 0.5                     # their share of the pixel / grounding loss
    mask_max: float = 0.9
    sprite_weight: float = 5.0
    drop_grid: float = 0.1
    drop_level: float = 0.2
    drop_status: float = 0.2
    w_recon: float = 1.0
    w_level: float = 10.0                        # grounding on z: level id + mode flags
    w_status: float = 10.0                       # grounding on z: tier, star, facing, lives

    # --- Dynamics: TSSM + prior head --------------------------------------------------------------------------------
    tssm_dim: int = 512
    tssm_layers: int = 6                         # TSSM blocks
    tssm_heads: int = 4
    tssm_kv_heads: int = 4
    tssm_ffn: int = 1365
    context_length: int = 64                     # attention window (steps)
    attn_gate: str = "elementwise"               # gated attention: "none", "headwise", "elementwise"
    prior_layers: int = 4

    # --- World-model losses -----------------------------------------------------------------------------------------
    w_dyn: float = 1.0                           # balanced KL with free nats (dyn : rep = 10 : 1)
    w_rep: float = 0.1
    free_nats: float = 1.0
    rollout_steps: int = 0                       # open-loop rollout on the model's own prior samples (0 = off)
    rollout_splits: int = 1
    rollout_decay: float = 0.9
    w_rollout: float = 1.0                       # (no effect with rollout_steps = 0)
    w_sparse_reward: float = 0.25                # sparse (level completion) reward head
    w_standard_reward: float = 0.5               # standard (coins, power-ups) reward head
    w_curiosity: float = 1.0                     # curiosity head (reads detached states: trains only itself)
    w_continue: float = 1.0                      # continue head (death-only with level_end_terminal = False)
    continue_zero_weight: float = 10.0
    continue_calibrated: bool = True             # p(alive) read with log(continue_zero_weight) added to the logit
    w_grid: float = 0.25                         # explored-grid head on s = [h, z] (its centre gates curiosity)
    grid_zero_weight: float = 5.0                # explored grid: unexplored cells up-weighted
    curiosity_grad: bool = False                 # curiosity head reads detached states
    w_dreamlike: float = 0.5                     # reward / continue / curiosity / grid heads: share on rollout states
    harmony: bool = False                         # learned loss weights (HarmonyDream);
    harmony_lr: float = 1e-3

    # --- Optimisation -----------------------------------------------------------------------------------------------
    wm_lr: float = 2e-4
    clip_percentile: float = 99.0
    clip_window: int = 2000
    clip_warmup: int = 500
    clip_initial: float = 1000.0
    mixed_precision: str = "auto"                # bf16 on RTX 30xx+, fp16 + loss scaling on older GPUs

    # --- Batches and replay -----------------------------------------------------------------------------------------
    batch_size: int = 96                       # sequences per update
    seq_len: int = 96                            # steps per sequence
    buffer_size: int = 500000                    # ~21 KB per transition
    death_sample_prob: float = 0.15               # windows that contain a death (at a uniform index)
    recent_sample_prob: float = 0.3
    reward_sample_prob: float = 0.025
    reward_threshold: float = 50.0
    level_end_terminal: bool = False             # True: the frame after a level completion is stored with cont = 0
    loss_sample_prob: float = 0.1                # windows centred on surprising frames
    loss_priority_alpha: float = 1.0

    # --- Imagination and behaviour ----------------------------------------------------------------------------------
    mlp_dim: int = 1024
    dream_horizon: int = 25
    dreams_per_sequence: int = 16
    dream_unimix: float = 0.0                    # dreams sample the raw prior
    dream_cuda_graph: bool = True
    death_dream_prob: float = 0.3                # P(a dream starts 5-10 steps before a death)
    death_dream_window: tuple = (2, 20)          # steps before the death
    dream_loss_frac: float = 0.05                 # dreams starting at surprising states
    branch_groups: int = 4                       # start states that try every first action
    curiosity_scale: float = 0.25                # weight of the curiosity advantage on the std-normalised streams (ext 1, cur this)
    death_penalty: float = 0.0                   # reward of a death (0 = off): dreams penalty x (1 - p(alive)), real deaths
    continue_discount: float = 0.997
    entropy_scale: float = 0.005                 # REINFORCE entropy bonus
    adv_norm: str = "std"                        # the only mode: each advantage stream centred and divided by its own std, then ext + curiosity_scale x cur
    value_target: str = "hl_gauss"               # critic / reward-head targets: "hl_gauss" or "twohot"
    hl_sigma_ratio: float = 0.75                 # HL-Gauss: sigma = this x bin width (symlog units)
    critic_ema_tau: float = 0.95
    w_repval: float = 0.3                        # replay value loss
    play_epsilon: float = 0.0                    # random actions while playing
    play_unimix: bool = False                    # 1% uniform mix of the actor while playing (training keeps it)
    actor_lr: float = 4e-5
    critic_lr: float = 1e-4

    # --- Speed (each flag computes the same result as with it off) ---------------------------------------------------
    compile: bool = True                         # torch.compile: encoder layers, decoder chunks, MLP heads, TSSM blocks
    act_cuda_graph: bool = True                  # acting step (encoder + actor + TSSM step) as one CUDA graph
    dec_subpixel: bool = True                    # decoder: nearest x3 + conv == low-res conv + PixelShuffle (same weights)
    dec_sac: bool = True                         # decoder checkpoint keeps conv outputs (~1.5 MB/frame bf16)
    dec_targets_once: bool = True                # pixel targets + entropy floor once per update, outside the checkpoint
    enc_fast_attn: bool = True                   # encoder attention: SDPA on views (same params, no MHA copies)
    skip_pad_frames: bool = True                 # posteriors of real frames only (pads copy their row's last frame)
    sync_free: bool = True                       # fused AdamW / Adam, GPU-side clipping, no host syncs per update
    h2d_stream: bool = True                      # the prefetcher copies batches to the GPU on a side stream
    async_save: bool = True                      # checkpoint + replay buffer written by a background thread
    async_dream_pdf: bool = True                 # dream PDFs rendered by a worker process

    # --- Diagnostics ------------------------------------------------------------------------------------------------
    drift_interval: int = 1000
    drift_frames: int = 256
    grad_share_interval: int = 500               # updates between per-term gradient norms on TSSM / encoder (0 = off)
    audit: bool = True                           # dream audit against the real game (False: no snapshots in play)
    audit_every: int = 5                         # rounds between audits (0 = off); only the play before one snapshots
    audit_starts: int = 50                       # start states per audit (30 % pre-death, the rest uniform)
    audit_horizon: int = 25                      # steps of each dream / real rollout (= dream_horizon)
    audit_snapshot_every: int = 40               # agent steps between the env snapshots taken in play

    # --- Run control and outputs (read by the Trainer) -----------------------------------------------------------
    total_rounds: int = 10000
    updates_per_round: int = 100
    initial_episodes_per_env: int = 3            # first collection with the untrained actor
    seed: int = 7
    checkpoint_interval: int = 5                 # rounds
    max_checkpoints: int = 15
    num_best_dreams: int = 3                     # dreams per round in the PDF
    num_random_dreams: int = 15
    csv_extra_columns: bool = False              # also log the diagnostic columns (EXTRA_COLUMNS) in the CSV
    reward_plot_interval: int = 5               # rounds
    reward_ma_window: int = 20                   # records in the plots' moving average
    death_map_interval: int = 20                 # rounds between death maps
    death_map_workers: int = 0                   # its replay processes; 0 = CPU count - 2 (max 8)
    num_eval_envs: int = 5                       # eval: SDL2 windows (recording: one MP4 each)
    num_eval_episodes: int = 1                   # eval: episodes per env


GQ_CONFIG = GQDreamerConfig()

MODEL_CONFIGS = {"gq": GQ_CONFIG}

# config keys of pre-GQ-Dreamer checkpoints -> current names
LEGACY_CONFIG_KEYS = {"tok_dim": "enc_dim", "tok_depth": "enc_depth", "tok_heads": "enc_heads",
                      "number_of_sequences": "batch_size", "steps_per_sequence": "seq_len",
                      "model_config": "config_name", "pmpo_weighting": "actor_objective"}

# renamed configurations -> current names
LEGACY_CONFIG_NAMES = {"gqfsq": "gq"}


def _legacy_values(d):
    """Maps renamed config keys and values to the current ones and drops keys that no longer exist."""
    d = dict(d)
    if "w_reward" in d:
        w = d.pop("w_reward")
        d.setdefault("w_sparse_reward", w)
        d.setdefault("w_standard_reward", w)
    if "reinforce_entropy_scale" in d:
        v = d.pop("reinforce_entropy_scale")
        if d.get("actor_objective", "reinforce") == "reinforce":
            d["entropy_scale"] = v
    for k in [k for k in d if k == "actor_objective" or k.startswith("pmpo_")]:
        del d[k]
    d.pop("adv_norm_decay", None)                                          # not a config key
    if d.get("adv_norm", "std") != "std":                                  # only per-stream std normalisation
        del d["adv_norm"]
    return d


def model_config(name="gq", **overrides):
    """Named configuration with optional field overrides."""
    name = LEGACY_CONFIG_NAMES.get(name, name)
    if name not in MODEL_CONFIGS:
        raise ValueError(f"unknown model configuration {name!r}; available: {sorted(MODEL_CONFIGS)}")
    return replace(MODEL_CONFIGS[name], **_legacy_values({LEGACY_CONFIG_KEYS.get(k, k): v
                                                          for k, v in overrides.items()}))


def config_from_dict(d, base=GQ_CONFIG):
    """Config from a dict (e.g. a checkpoint's); unknown keys are ignored."""
    names = {f.name for f in fields(GQDreamerConfig)}
    d = _legacy_values({LEGACY_CONFIG_KEYS.get(k, k): v for k, v in d.items()})
    if d.get("config_name") in LEGACY_CONFIG_NAMES:
        d["config_name"] = LEGACY_CONFIG_NAMES[d["config_name"]]
    d = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in names}
    return replace(base, **d)


def config_to_dict(cfg):
    """Plain dict of a config (stored in checkpoints)."""
    return asdict(cfg)

