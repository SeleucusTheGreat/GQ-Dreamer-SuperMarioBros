"""Checkpointing of GQ-Dreamer: save, find and partially load checkpoints (every part whose names and shapes match)."""
import glob
import os
import threading

import torch

from .config import LEGACY_CONFIG_KEYS, LEGACY_CONFIG_NAMES

# pre-GQ-Dreamer part names -> current names
LEGACY_NAMES = {
    "sparseRewardPredictor": "sparse_reward_head", "standardRewardPredictor": "standard_reward_head",
    "curiosityPredictor": "curiosity_head", "continuePredictor": "continue_head",
    "worldModelOptimizer": "wm_optimizer", "actorOptimizer": "actor_optimizer", "criticOptimizer": "critic_optimizer",
    "curiosityCriticOptimizer": "curiosity_critic_optimizer", "harmonyOptimizer": "harmony_optimizer",
}


def _cpu_copy(obj):
    """Nested dicts / lists of tensors -> the same structure with CPU copies of the tensors."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):                                          # keeps OrderedDict and state_dict metadata
        out = type(obj)((k, _cpu_copy(v)) for k, v in obj.items())
        if hasattr(obj, "_metadata"):
            out._metadata = obj._metadata
        return out
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu_copy(v) for v in obj)
    return obj


def _short_list(keys, n=4):
    """First n keys, then a count of the rest."""
    keys = list(keys)
    return ", ".join(keys[:n]) + (f", ... (+{len(keys) - n})" if len(keys) > n else "")


def migrate_legacy_checkpoint(ckpt):
    """Maps renamed parts, config keys and configuration names of a checkpoint to the current ones."""
    ckpt = _migrate_pre_gq_dreamer(ckpt)
    cfg = ckpt.get("config")
    if isinstance(cfg, dict) and cfg.get("config_name") in LEGACY_CONFIG_NAMES:
        cfg["config_name"] = LEGACY_CONFIG_NAMES[cfg["config_name"]]
    return ckpt


def _migrate_pre_gq_dreamer(ckpt):
    """Maps a pre-GQ-Dreamer checkpoint to current names; folds the unused noise embedding into the token bias."""
    if not any(k in ckpt for k in LEGACY_NAMES):
        return ckpt
    for old, new in LEGACY_NAMES.items():
        if old in ckpt:
            ckpt[new] = ckpt.pop(old)

    def rename(param):
        head, _, rest = param.partition(".")
        return f"{LEGACY_NAMES.get(head, head)}.{rest}"
    ckpt["optimizer_param_names"] = {LEGACY_NAMES.get(opt, opt): [rename(p) for p in names]
                                     for opt, names in ckpt.get("optimizer_param_names", {}).items()}
    ckpt["config"] = {LEGACY_CONFIG_KEYS.get(k, k): v for k, v in ckpt.get("config", {}).items()}
    tssm = ckpt.get("tssm", {})
    if "noise_in.weight" in tssm:                                      # fold the noise embedding's bucket 0 into the bias
        tssm["token_mlp.2.bias"] = tssm["token_mlp.2.bias"] + tssm.pop("noise_in.weight")[0].to(
            tssm["token_mlp.2.bias"].dtype)
    return ckpt


class CheckpointMixin:
    """Save / load of every GQDreamer module, optimizer and run counter."""
    CHECKPOINT_DIR = "checkpoints"
    CHECKPOINT_FAMILY = "gq_dreamer"
    ARCH_VERSION = "v8"            # architecture id in checkpoint and buffer file names
    MODULES = ("tokenizer", "tssm", "prior", "sparse_reward_head", "standard_reward_head", "curiosity_head",
               "continue_head", "grid_head", "actor", "critic", "curiosity_critic", "critic_target",
               "curiosity_critic_target", "harmony")
    OPTIMIZERS = ("wm_optimizer", "actor_optimizer", "critic_optimizer", "curiosity_critic_optimizer",
                  "harmony_optimizer")
    OPTIMIZER_MODULES = {                                              # parameter order of each optimizer
        "wm_optimizer": ("tokenizer", "tssm", "prior", "sparse_reward_head", "standard_reward_head",
                         "curiosity_head", "continue_head", "grid_head"),
        "actor_optimizer": ("actor",),
        "critic_optimizer": ("critic",),
        "curiosity_critic_optimizer": ("curiosity_critic",),
        "harmony_optimizer": ("harmony",),
    }
    ACTOR_CRITIC_PARTS = ("actor", "critic", "curiosity_critic", "critic_target", "curiosity_critic_target",
                          "actor_optimizer", "critic_optimizer", "curiosity_critic_optimizer")
    ARCH_KEYS = ("arch_version", "action_dim", "n_latents", "classes", "enc_stem", "enc_patch_tiles", "enc_input",
                 "enc_dim", "enc_depth", "enc_heads", "enc_pos_inject", "enc_swiglu", "enc_ffn", "decoder", "dec_dim",
                 "dec_depth", "dec_heads", "dec_patch_tiles", "dec_chans", "head_hidden", "tssm_dim", "tssm_layers",
                 "tssm_heads", "tssm_kv_heads", "tssm_ffn", "prior_layers", "mlp_dim", "attn_gate")
    COMPAT_KEYS = ("frames_per_step", "config_name")                   # a mismatch blocks resuming

    # --- Names ------------------------------------------------------------------------------------------------------
    def _set_file_names(self):
        """Checkpoint / buffer file names of this configuration and control rate (the same for either decoder)."""
        tag = "" if self.cfg.config_name == "gq" else f"_{self.cfg.config_name}"
        self.BUFFER_NAME = f"replay_buffer_{self.CHECKPOINT_FAMILY}_{self.ARCH_VERSION}{tag}.buffer"
        self.CHECKPOINT_PREFIX = f"{self.CHECKPOINT_FAMILY}_{self.ARCH_VERSION}{tag}_f{self.cfg.frames_per_step}"
        self.CHECKPOINT_GLOB = f"{self.CHECKPOINT_PREFIX}_R*_G*.pt"

    def optimizer_named_params(self, name):
        """[(qualified name, parameter)] updated by optimizer `name`."""
        return [(f"{m}.{n}", p) for m in self.OPTIMIZER_MODULES[name] for n, p in getattr(self, m).named_parameters()]

    def optimizer_params(self, name):
        """Parameters updated by optimizer `name`."""
        return [p for _, p in self.optimizer_named_params(name)]

    def find_checkpoint(self):
        """Newest checkpoint (highest gradient step) of this configuration, or None."""
        files = glob.glob(os.path.join(self.CHECKPOINT_DIR, self.CHECKPOINT_GLOB))
        if not files:
            return None
        return max(files, key=lambda x: int(x.split("_G")[-1].split(".pt")[0]))

    # --- Save -------------------------------------------------------------------------------------------------------
    def save_checkpoint(self, path, save_buffer=True, background=False, after=None):
        """Writes modules, optimizers and counters to `path` (and the replay buffer next to it). background: the state
        is copied to CPU memory here and a thread writes the files (returned: join it before the replay buffer
        changes, i.e. before the next play); after: called once the files are written."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        ckpt = {name: getattr(self, name).state_dict() for name in self.MODULES + self.OPTIMIZERS}
        ckpt.update({
            "wm_clipper": self.wm_clipper.state_dict(),
            "total_num_episodes": self.total_num_episodes,
            "total_num_steps": self.total_num_steps,
            "total_num_updates": self.total_num_updates,
            "config": self.config,
            "optimizer_param_names": {name: [n for n, _ in self.optimizer_named_params(name)]
                                      for name in self.OPTIMIZERS},
        })
        if not background:
            self._write_checkpoint(ckpt, path, save_buffer, after)
            return None
        ckpt = _cpu_copy(ckpt)                                         # snapshot: training goes on meanwhile
        thread = threading.Thread(target=self._write_checkpoint, args=(ckpt, path, save_buffer, after),
                                  name="checkpoint-writer")
        thread.start()
        return thread

    def _write_checkpoint(self, ckpt, path, save_buffer, after):
        """Writes a checkpoint dict (+ the replay buffer); errors are reported, not raised."""
        try:
            torch.save(ckpt, path)
            print(f"\033[90m    [ckpt] saved {path}{' (+ replay buffer)' if save_buffer else ''}\033[0m")
            if save_buffer:
                self.buffer.save(os.path.join(os.path.dirname(path) or ".", self.BUFFER_NAME))
            if after is not None:
                after()
        except Exception as e:
            print(f"[!] Warning: checkpoint {path} not written completely: {type(e).__name__}: {e}")

    # --- Load -------------------------------------------------------------------------------------------------------
    @staticmethod
    def _load_module_partial(name, module, saved):
        """Loads every tensor whose name and shape match; the rest keeps its fresh init. -> True if all matched."""
        own = module.state_dict()
        fits = {k: v for k, v in saved.items() if k in own and own[k].shape == v.shape}
        reshaped = [k for k, v in saved.items() if k in own and own[k].shape != v.shape]
        new = [k for k in own if k not in saved]
        dropped = [k for k in saved if k not in own]
        module.load_state_dict(fits, strict=False)
        if not (reshaped or new or dropped):
            return True
        msg = f"[load] '{name}': {len(fits)}/{len(own)} tensors restored"
        if reshaped:
            msg += f" | shape changed -> fresh: {_short_list(reshaped)}"
        if new:
            msg += f" | new -> fresh: {_short_list(new)}"
        if dropped:
            msg += f" | no longer in the model -> dropped: {_short_list(dropped)}"
        print(msg)
        return False

    def _load_optimizer_partial(self, name, ckpt):
        """Restores optimizer moments by parameter name (same shape); others start fresh."""
        opt = getattr(self, name)
        saved = ckpt[name]
        old_idx = [i for g in saved["param_groups"] for i in g["params"]]
        old_state = saved.get("state", {})
        names = ckpt.get("optimizer_param_names", {}).get(name)
        if names is None or len(names) != len(old_idx):
            print(f"[load] '{name}': no usable parameter names -> fresh optimizer state")
            return
        by_name = {n: old_state[i] for n, i in zip(names, old_idx) if i in old_state}
        current = self.optimizer_named_params(name)
        new_state = {}
        for idx, (n, p) in enumerate(current):
            st = by_name.get(n)
            if st is not None and all(tuple(t.shape) == tuple(p.shape)
                                      for t in st.values() if torch.is_tensor(t) and t.dim() > 0):
                new_state[idx] = st
        sd = opt.state_dict()
        sd["state"] = new_state
        opt.load_state_dict(sd)
        if old_state and len(new_state) < len(current):                 # (an optimizer never stepped has none)
            print(f"[load] '{name}': moments restored for {len(new_state)}/{len(current)} parameters "
                  f"(the others start fresh)")

    def load_checkpoint(self, path=None, reset_actor_critic=False, load_buffer=True):
        """Loads the newest (or given) checkpoint part by part; reset_actor_critic keeps only the world model."""
        if path is None:
            path = self.find_checkpoint()
            if path is None:
                print(f"\033[90m    [setup] no {self.CHECKPOINT_PREFIX} checkpoint in {self.CHECKPOINT_DIR}/ -- the "
                      f"model starts from scratch\033[0m")
                if load_buffer:
                    self._load_buffer(self.CHECKPOINT_DIR)
                return
        if not os.path.exists(path):
            print(f"\033[90m    [setup] no checkpoint found at {path} -- starting from scratch\033[0m")
            return
        print(f"\033[90m    [setup] loading checkpoint {path}\033[0m")
        ckpt = migrate_legacy_checkpoint(torch.load(path, map_location="cpu", weights_only=False))   # not via GPU
        old_cfg, new_cfg = ckpt.get("config", {}), self.config

        def diff(keys):
            return {k: (old_cfg.get(k), new_cfg.get(k)) for k in keys if old_cfg.get(k) != new_cfg.get(k)}

        def fmt(d):
            return ", ".join(f"{k}: {a} -> {b}" for k, (a, b) in d.items())

        incompatible = diff(self.COMPAT_KEYS)
        if incompatible:
            print(f"[load] {os.path.basename(path)} was trained with {fmt(incompatible)}: not loaded, the model "
                  f"starts from scratch (the replay buffer is still reused if it matches).")
            del ckpt
            if load_buffer:
                self._load_buffer(os.path.dirname(path) or ".")
            return
        changed = diff(self.ARCH_KEYS)
        if changed:
            print(f"[load] architecture changed since {os.path.basename(path)} ({fmt(changed)}): loading every part "
                  f"that still fits, the rest starts fresh.")
        if reset_actor_critic:
            print("[load] RESET_ACTOR_CRITIC: actor, critics and their optimizers start fresh.")

        skip = set(self.ACTOR_CRITIC_PARTS) if reset_actor_critic else set()
        exact = True
        for name in self.MODULES:
            if name in skip:
                continue
            if name not in ckpt:
                print(f"[load] '{name}' not in checkpoint -> fresh init.")
                exact = False
                continue
            exact &= self._load_module_partial(name, getattr(self, name), ckpt[name])
        if reset_actor_critic:
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.curiosity_critic_target.load_state_dict(self.curiosity_critic.state_dict())
        for name in self.OPTIMIZERS:
            if name in skip or name not in ckpt:
                continue
            try:
                self._load_optimizer_partial(name, ckpt)
            except Exception as e:                                     # optimizer state never blocks a resume
                print(f"[load] '{name}' could not be restored -> fresh optimizer state ({e}).")

        self.wm_clipper.load_state_dict(ckpt.get("wm_clipper", {}))
        self.total_num_episodes = ckpt.get("total_num_episodes", 0)
        self.total_num_steps = ckpt.get("total_num_steps", 0)
        self.total_num_updates = ckpt.get("total_num_updates", 0)
        self._resumed = True
        if exact:
            print("\033[90m    [setup] checkpoint restored completely\033[0m")
        del ckpt
        if load_buffer:
            self._load_buffer(os.path.dirname(path) or ".")

    def _load_buffer(self, directory):
        """Loads the replay buffer file of this configuration if it exists."""
        path = os.path.join(directory, self.BUFFER_NAME)
        if os.path.exists(path):
            self.buffer.load(path)
