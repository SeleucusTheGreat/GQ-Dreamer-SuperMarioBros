"""Training / evaluation loop of GQ-Dreamer: rounds of updates and play, CSV metrics, plots, dream PDFs, checkpoints,
and the notebook entry point run()."""
import csv
import gc
import glob
import multiprocessing as mp
import os
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from .gq_dreamer import GQDreamer
from .audit import AUDIT_COLUMNS, audit_enabled, audit_lines
from .config import model_config
from .death_maps import DeathMapJob
from .env import create_envs
from .game import FRAMES_PER_STEP, MAIN_LEVELS
from .plots import find_comparison_runs, plot_reward_comparison, plot_training
from .visualize import save_dreams_pdf

# ANSI styles of the console output
_B, _DIM, _CYA, _GRN = "\033[1m", "\033[90m", "\033[96m", "\033[92m"
_MAG, _YEL, _RST = "\033[95m", "\033[93m", "\033[0m"

RECORD_FPS = 60                                        # eval videos: every emulator frame (GBC ~59.7 Hz), real time


def seed_everything(seed):
    """Seeds python, numpy and torch (cuDNN autotuning stays on)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _dreams_to_cpu(dreams):
    """Packed dreams (GPU tensors with sync_free) -> float metric + CPU tensors."""
    return [(float(d[0]),) + tuple(x.cpu() if torch.is_tensor(x) else x for x in d[1:-1]) + (d[-1],) for d in dreams]


class Trainer:
    """Runs GQ-Dreamer: alternates world-model / imagination updates with play rounds, logs and checkpoints."""

    FULL_SUMMARY = False                     # round summary: False = header, world model KL and speed; True = every row

    # CSV column -> (metrics source, metric key)
    COLUMNS = [
        ("bestReward", "play", "best_reward"), ("meanX", "play", "mean_x"), ("episodeLength", "play", "episode_length"),
        ("worldModelLoss", "wm", "world_model_loss"),          # round mean; = the 10 wm* columns below + curiosity
        ("wmPixel", "wm", "wm_pixel"), ("wmLevel", "wm", "wm_level"), ("wmStatus", "wm", "wm_status"),
        ("wmKlDyn", "wm", "wm_kl_dyn"), ("wmKlRep", "wm", "wm_kl_rep"), ("wmRollout", "wm", "wm_rollout"),
        ("wmSparseReward", "wm", "wm_sparse_reward"), ("wmStandardReward", "wm", "wm_standard_reward"),
        ("wmContinue", "wm", "wm_continue"), ("wmGrid", "wm", "wm_grid"),   # (curiosity term: console only)
        ("reconLoss", "wm", "recon_excess"),                  # pixel BCE minus its entropy floor (0 = perfect)
        ("klLoss", "wm", "kl_dyn"),
        ("rewardLoss", "wm", "reward_loss"),
        ("continueLoss", "wm", "continue_loss"),
        ("pAliveAtDeath", "wm", "p_alive_at_death"),
        ("pAliveDeathPred", "wm", "p_alive_death_pred1"),     # death probe, round average
        ("pAliveDeathDream", "wm", "p_alive_death_dream"),
        ("dreamDeathSurvival", "wm", "dream_death_survival"),
        ("actorLoss", "dream", "actor_loss"),
        ("criticLoss", "dream", "critic_loss"),
        ("entropies", "dream", "entropies"),
        ("advantages", "dream", "advantages"),
        ("criticValues", "dream", "critic_values"),
        ("extCriticEV", "dream", "ext_critic_ev"), ("curCriticEV", "dream", "cur_critic_ev"),   # critics learning?
        ("extAdvShare", "dream", "ext_adv_share"), ("curAdvShare", "dream", "cur_adv_share"),   # who drives the actor
        ("deathPenalty", "play", "death_penalty"),             # per episode (not in totalReward)
        ("furthestLevel", "play", "furthest_level"),           # index into game.MAIN_LEVELS (-1: none); plot shading
        ("nonplayableSteps", "play", "nonplayable_steps"),     # share of steps on the map / transition screens
    ]                                                          # the dream audit has its own CSV (audit_filename)
    GRAD_SHARE_COLUMNS = [(f"g{'Tssm' if g == 'tssm' else 'Enc'}Share{''.join(w.title() for w in t.split('_'))}",
                           "wm", f"gshare_{g}_{t}")
                          for g in GQDreamer.GRAD_SHARE_GROUPS for t in GQDreamer.WM_LOSS_TERMS]
    EXTRA_COLUMNS = [                                          # written only with csv_extra_columns = True
        ("totalCuriosity", "play", "total_curiosity"), ("furthestX", "play", "furthest_x"),
        ("reconSpriteLoss", "wm", "recon_sprite_excess"), ("rolloutLoss", "wm", "rollout_loss"),
        ("encoderDrift", "wm", "encoder_drift"), ("curiosityAdvantages", "dream", "curiosity_advantages"),
        ("replayReturns", "dream", "replay_returns"),
        ("secPerUpdate", "time", "per_update"), ("playSeconds", "play", "t_play"),
        ("reconBranchLoss", "wm", "recon_loss"), ("maskRatio", "wm", "mask_ratio"),
        ("postEntropy", "wm", "post_entropy"), ("priorEntropy", "wm", "prior_entropy"),
        ("latentUsage", "wm", "latent_usage"), ("predMatch", "wm", "pred_match"), ("copyMatch", "wm", "copy_match"),
        ("rolloutMatchK", "wm", "rollout_match_k"), ("rolloutKlK", "wm", "rollout_kl_k"),
        ("gridLoss", "wm", "grid_loss"), ("levelLoss", "wm", "level_loss"),
        ("statusLoss", "wm", "status_loss"), ("levelAcc", "wm", "level_acc"), ("tierAcc", "wm", "tier_acc"),
        ("levelAccNoToken", "wm", "level_acc_notoken"), ("gridAcc", "wm", "grid_acc"),
        ("gridCenterAcc", "wm", "grid_center_acc"), ("groundingLoss", "wm", "grounding_loss"),
        ("dreamLevelAcc", "wm", "dream_level_acc"), ("dreamTierAcc", "wm", "dream_tier_acc"),
        ("curiosityLoss", "wm", "curiosity_loss"),
        ("advScale", "dream", "adv_scale"), ("curAdvScale", "dream", "cur_adv_scale"),
        ("dreamStartPreDeath", "dream", "dream_start_pre_death"),
        ("advAbsMean", "dream", "adv_abs_mean"),
        ("branchAgree", "dream", "branch_agree"), ("branchSpread", "dream", "branch_spread"),
        ("preDeathStates", "wm", "pre_death_states"), ("harmonizedLoss", "wm", "harmonized_loss"),
        ("hwPixel", "wm", "harmony_pixel"), ("hwGrounding", "wm", "harmony_grounding"), ("hwKl", "wm", "harmony_kl"),
        ("hwRollout", "wm", "harmony_rollout"), ("hwReward", "wm", "harmony_reward"), ("hwCont", "wm", "harmony_cont"),
        ("hwGrid", "wm", "harmony_grid"),
        ("wmGradNorm", "wm", "wm_grad_norm"), ("wmClipThreshold", "wm", "wm_clip_threshold"),
        ("wmClipFraction", "wm", "wm_clip_fraction"), ("pAliveAlive", "wm", "p_alive_alive"),
        ("paddedFraction", "wm", "padded_fraction"),
        ("curiosityCriticLoss", "dream", "curiosity_critic_loss"),
        ("targetCriticValues", "dream", "target_critic_values"),
        ("curiosityCriticValues", "dream", "curiosity_critic_values"),
        ("advantagePosFraction", "dream", "advantage_pos_fraction"),
        ("curiosityFlipFraction", "dream", "curiosity_flip_fraction"),
        ("dreamMeanCuriosity", "dream", "dream_mean_curiosity"), ("dreamMeanAlive", "dream", "dream_mean_alive"),
        ("dreamDeathFraction", "dream", "dream_death_fraction"),
        ("replayValueLoss", "dream", "replay_value_loss"), ("replayCriticValues", "dream", "replay_critic_values"),
        ("deathRate", "play", "deaths"), ("timeoutRate", "play", "timeouts"),
        ("bestCuriosity", "play", "best_curiosity"),
        ("shortEpisodes", "play", "short_episodes"), ("trainSeconds", "time", "train"),
        ("batchWaitSeconds", "time", "wait"),
        ("gpuPeakGB", "mem", "gpu_peak_gb"), ("gpuReservedGB", "mem", "gpu_reserved_gb"),
        ("gpuFreeGB", "mem", "gpu_free_gb"),
        ("advCombinedStd", "dream", "adv_combined_std"), ("dreamDeathPenalty", "dream", "dream_death_penalty"),
        ("extAdvStd", "dream", "ext_adv_std"), ("curAdvStd", "dream", "cur_adv_std"),
        ("extReturnRange", "dream", "ext_return_range"), ("curReturnRange", "dream", "cur_return_range"),
        ("skippedSteps", "play", "skipped_steps"), ("gradShareUpdate", "wm", "grad_share_update"),
    ] + GRAD_SHARE_COLUMNS
    BASE_HEADERS = ["envSteps", "gradientSteps", "totalReward"]

    def __init__(self, envs, device=None, config="gq", visualize_dreams=False, verbose=True, **overrides):
        """config: a name of MODEL_CONFIGS ("gq"); overrides: GQDreamerConfig fields changed on top of it."""
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.envs = envs
        self.visualize_dreams = visualize_dreams
        sfx = f"_{config}"                             # own CSV / plots / dreams, next to GQ-DreamerOld's

        cfg = model_config(config, action_dim=envs.action_space.n,
                           frames_per_step=getattr(envs, "frames_per_step", FRAMES_PER_STEP), **overrides)

        # run control and outputs: set in GQDreamerConfig (config.py), overridable via run(...) / Trainer(...)
        for name in ("total_rounds", "updates_per_round", "initial_episodes_per_env", "seed", "checkpoint_interval",
                     "max_checkpoints", "num_best_dreams", "num_random_dreams",
                     "csv_extra_columns", "reward_plot_interval", "reward_ma_window", "death_map_interval",
                     "death_map_workers", "num_eval_envs", "num_eval_episodes"):
            setattr(self, name, getattr(cfg, name))
        self.csv_filename = "metrics/training_data.csv"   # the same files with either decoder
        self.audit_filename = "diagnostics/dream_audit.csv"   # dream audit rows (only rounds that ran one)
        self.reward_plot_filename = f"plots/rewards{sfx}.png"
        self.reward_compare_filename = f"plots/reward_compare{sfx}.png"        # over gradient updates
        self.dreams_dir = "dreams"
        self.reward_compare_labels = {}                # comparison plot: legend names of other metrics/*.csv files

        seed_everything(self.seed)
        self.agent = GQDreamer(cfg, device=self.device, envs=envs, verbose=verbose)
        self._save_thread = None                       # background checkpoint writer (async_save)
        self._pdf_pool = None                          # dream PDF worker process (async_dream_pdf)
        self._pdf_async_ok = True
        self._death_maps = DeathMapJob(config=cfg.config_name, workers=self.death_map_workers or None)

    @property
    def columns(self):
        """Logged (header, source, key) columns."""
        return self.COLUMNS + (self.EXTRA_COLUMNS if self.csv_extra_columns else [])

    @property
    def headers(self):
        """CSV header row."""
        return self.BASE_HEADERS + [c[0] for c in self.columns]

    # --- Training ---------------------------------------------------------------------------------------------------
    def train(self, reset_actor_critic=False):
        """Main loop; reset_actor_critic keeps world model and buffer but restarts actor and critics."""
        agent = self.agent
        agent.load_checkpoint(reset_actor_critic=reset_actor_critic)
        self._prepare_csv()
        self._update_plots()

        if len(agent.buffer) == 0:
            n0 = self.initial_episodes_per_env
            print(f"{_DIM}    [setup] replay buffer empty -- collecting {n0} episode(s) per env...{_RST}")
            s, c = agent.play(episodes_per_env=n0, snapshots=self._audit_due(0))
            print(f"{_DIM}    [setup] initial collection | reward {s} | curiosity {c}{_RST}")

        try:
            for rnd in range(self.total_rounds):
                self._train_and_play_round(rnd)
        finally:
            self._wait_for_save()
            if self._pdf_pool is not None:
                self._pdf_pool.shutdown(wait=True)
                self._pdf_pool = None
            self._report_death_map()
            if self._death_maps.busy:
                print(f"{_DIM}    [death map] G{self._death_maps.update} is still being drawn by its worker process; "
                      f"it will appear in diagnostics/images/{_RST}")
            self._death_maps.close(wait=False)

    def _train_and_play_round(self, rnd):
        """One round: updates, dream PDF, play, logging, plots and checkpoint."""
        agent = self.agent
        self._print_round_header(rnd)
        self._report_death_map()
        wm_metrics, dream_metrics, timing, best_dreams, random_dreams = self._train_round()
        audit = self._audit(rnd) if self._audit_due(rnd) else {}

        if self.visualize_dreams and best_dreams:
            best_dreams.sort(key=lambda x: x[0], reverse=True)
            vis = best_dreams[:self.num_best_dreams] + random.sample(
                random_dreams, min(self.num_random_dreams, len(random_dreams)))
            self._check_pdf_worker()
            path = save_dreams_pdf(agent, vis, self.dreams_dir, executor=self._dream_pdf_pool())
            how = " (drawn by a worker process)" if self._pdf_pool is not None else ""
            print(f"{_DIM}    [dreams] saved {len(vis)} dreams to {path}{how}{_RST}")

        self._wait_for_save()                                      # the buffer changes while playing
        print(f"{_DIM}    [play] running 1 episode per env with the updated policy...{_RST}")
        avg_score, avg_curiosity = agent.play(episodes_per_env=1, snapshots=self._audit_due(rnd + 1))
        play = dict(agent.last_play_stats, total_curiosity=avg_curiosity)
        mem = self._gpu_memory()
        self._print_summary(avg_score, avg_curiosity, wm_metrics, dream_metrics, play, timing, mem, audit)
        self._append_csv_row(avg_score, {"wm": wm_metrics, "dream": dream_metrics, "play": play, "time": timing,
                                         "mem": mem})
        if audit:
            self._append_audit_row(audit)
        if self.death_map_interval > 0 and (rnd + 1) % self.death_map_interval == 0:
            self._start_death_map()

        if (rnd + 1) % self.reward_plot_interval == 0:
            self._update_plots()
        if rnd > 0 and rnd % self.checkpoint_interval == 0:
            self._save_checkpoint(avg_score)

    def _audit_due(self, rnd):
        """True when round `rnd` (of this session) runs the dream audit (every audit_every rounds: 0, N, 2N, ...); the
        play before it takes the snapshots."""
        c = self.agent.cfg
        return audit_enabled(c) and rnd % int(c.audit_every) == 0

    def _audit(self, rnd):
        """Runs the dream audit after the round's updates (before its play) -> metrics; a failure is reported and
        training goes on."""
        try:
            m = self.agent.audit()
            if not m:
                print(f"{_DIM}    [audit] no usable snapshot from the last play: skipped{_RST}")
            return m
        except Exception as e:                                         # a diagnostic never stops training
            print(f"[!] Warning: dream audit failed in round {rnd + 1}: {type(e).__name__}: {e}")
            return {}

    def _train_round(self):
        """updates_per_round world-model + imagination updates -> (wm metrics, dream metrics, timing, dreams)."""
        agent = self.agent
        self._free_gpu_cache()                                         # drop cached blocks left by play / plots
        best_dreams, random_dreams = [], []
        wm_metrics, dream_metrics = {}, {}
        n_updates, t_wait = 0, 0.0
        t_train = time.perf_counter()
        agent.start_prefetch()
        try:
            for step in range(self.updates_per_round):
                if step % 50 == 0:
                    print(f"{_DIM}    [train] step {step} / {self.updates_per_round}{_RST}")
                tw = time.perf_counter()
                batch = agent.sample_batch()
                t_wait += time.perf_counter() - tw
                if batch is None:
                    print("[!] Not enough data in the buffer yet; collecting more.")
                    break
                compute_metrics = step == self.updates_per_round - 1
                m, d, best, rnd = self._update(batch, compute_metrics)   # a CUDA OOM raises with its own traceback
                batch = None                                           # free it before the next batch arrives
                if compute_metrics:
                    wm_metrics, dream_metrics = m, d
                if self.visualize_dreams:
                    best_dreams.append(best)
                    random_dreams.append(rnd)
                agent.total_num_updates += 1
                n_updates += 1
        finally:
            agent.stop_prefetch()                                      # the buffer changes while playing
            agent.flush_priorities()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        best_dreams, random_dreams = _dreams_to_cpu(best_dreams), _dreams_to_cpu(random_dreams)
        t_train = time.perf_counter() - t_train
        timing = {"train": t_train, "per_update": t_train / max(n_updates, 1), "wait": t_wait,
                  "phases": dict(agent.timer.times)}
        return wm_metrics, dream_metrics, timing, best_dreams, random_dreams

    def _update(self, batch, compute_metrics):
        """One world-model + imagination update; its large tensors are freed on return."""
        agent = self.agent
        states, kv_context, start_mask, m, start_info = agent.train_world_model(batch, compute_metrics=compute_metrics)
        # real rewards / continues aligned with the states (state j = observation j + 1)
        real = {"rewards": agent.real_rewards(batch),               # + death penalty at the death rows
                "continues": batch["continues"][:, 1:], "valid": batch["valid"][:, 1:]}
        d, best, rnd = agent.imagine(states, kv_context, start_mask=start_mask, start_info=start_info, real=real,
                                     compute_metrics=compute_metrics, pack_dreams=self.visualize_dreams)
        return m, d, best, rnd

    # --- GPU memory --------------------------------------------------------------------------------------------------
    def _free_gpu_cache(self):
        """Returns unused cached GPU blocks to the driver (limits fragmentation between rounds)."""
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

    def _gpu_memory(self):
        """GPU memory in GB: allocated / reserved now, peak of the round, free on the device (all processes)."""
        if self.device.type != "cuda":
            return {}
        free, total = torch.cuda.mem_get_info(self.device)
        gb = 1024 ** 3
        return {"gpu_allocated_gb": torch.cuda.memory_allocated(self.device) / gb,
                "gpu_reserved_gb": torch.cuda.memory_reserved(self.device) / gb,
                "gpu_peak_gb": torch.cuda.max_memory_allocated(self.device) / gb,
                "gpu_free_gb": free / gb, "gpu_total_gb": total / gb}

    def _save_checkpoint(self, avg_score):
        """Saves a checkpoint (+ buffer) and keeps only the newest max_checkpoints (async_save: written by a background
        thread while the next round trains; joined before the next play)."""
        agent = self.agent
        agent.buffer.print_diagnostics()
        ckpt_dir = agent.CHECKPOINT_DIR
        path = os.path.join(ckpt_dir, f"{agent.CHECKPOINT_PREFIX}_R{int(avg_score)}_G{agent.total_num_updates}.pt")

        def prune():
            ckpts = sorted(glob.glob(os.path.join(ckpt_dir, agent.CHECKPOINT_GLOB)), key=os.path.getmtime)
            while len(ckpts) > self.max_checkpoints:
                oldest = ckpts.pop(0)
                try:
                    os.remove(oldest)
                except OSError as e:
                    print(f"[!] Warning: could not remove old checkpoint {oldest}: {e}")
        if agent.cfg.async_save:
            self._wait_for_save()
            self._save_thread = agent.save_checkpoint(path, background=True, after=prune)
            print(f"{_DIM}    [ckpt] writing {path} (+ replay buffer) in the background{_RST}")
        else:
            agent.save_checkpoint(path)
            prune()

    def _wait_for_save(self):
        """Waits for the background checkpoint writer (before the replay buffer changes)."""
        if self._save_thread is not None:
            if self._save_thread.is_alive():
                t = time.perf_counter()
                self._save_thread.join()
                print(f"{_DIM}    [ckpt] waited {time.perf_counter() - t:.1f}s for the checkpoint writer{_RST}")
            self._save_thread = None

    def _dream_pdf_pool(self):
        """The dream PDF worker process (async_dream_pdf), or None to draw in this process."""
        if not (self.agent.cfg.async_dream_pdf and self._pdf_async_ok):
            return None
        if self._pdf_pool is None:
            self._pdf_pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
        return self._pdf_pool

    def _check_pdf_worker(self):
        """Reports a failed PDF of the previous round (and falls back to drawing in this process)."""
        fut = save_dreams_pdf.pending
        if fut is not None and fut.done() and fut.exception() is not None:
            print(f"[!] dream PDF worker failed ({fut.exception()!r}); dream PDFs are drawn in this process from now on")
            self._pdf_pool.shutdown(wait=False)
            self._pdf_pool = None
            self._pdf_async_ok = False
        save_dreams_pdf.pending = None if fut is None or fut.done() else fut

    # --- Death maps -------------------------------------------------------------------------------------------------
    def _start_death_map(self):
        """Copies the replay buffer and draws its death map in a worker process (death_maps.py); never stops
        training."""
        job, g = self._death_maps, self.agent.total_num_updates
        try:
            if job.submit(self.agent.buffer, g):
                print(f"{_DIM}    [death map] drawing the replay buffer's deaths at G{g} in a worker process -> "
                      f"diagnostics/images/death_map_G{g}.png{_RST}")
            else:
                print(f"{_DIM}    [death map] G{job.update} is still being drawn: skipping G{g}{_RST}")
        except Exception as e:
            print(f"[!] Warning: could not start the death map: {e!r}")

    def _report_death_map(self):
        """Prints the summary of a death map finished since the last call (or why it failed)."""
        res = self._death_maps.poll()
        if res is None:
            return
        path, text = res
        if path is None:
            print(f"[!] Warning: death map G{self._death_maps.update} failed: {text}")
            return
        print(f"{_DIM}    [death map] -> {path}{_RST}")
        for line in text.splitlines():
            print(f"{_DIM}      {line}{_RST}")

    # --- Metrics CSV and plots --------------------------------------------------------------------------------------
    def _prepare_csv(self):
        """Creates the CSV, archives it for a fresh model, or resumes the step counters from it."""
        agent, path = self.agent, self.csv_filename
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path) and not agent._resumed:                # fresh model: archive the existing CSV
            with open(path, "r") as f:
                has_rows = sum(1 for line in f if line.strip()) > 1
            if has_rows:
                stem, ext = os.path.splitext(path)
                archived = f"{stem}_before_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
                os.replace(path, archived)
                print(f"{_DIM}    [setup] new model: previous CSV moved to {archived}{_RST}")
        apath = self.audit_filename
        if os.path.exists(apath) and not agent._resumed:
            with open(apath, "r") as f:
                has_rows = sum(1 for line in f if line.strip()) > 1
            if has_rows:
                stem, ext = os.path.splitext(apath)
                archived = f"{stem}_before_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
                os.replace(apath, archived)
                print(f"{_DIM}    [setup] new model: previous audit CSV moved to {archived}{_RST}")
        if not os.path.exists(path):
            with open(path, mode="w", newline="") as f:
                csv.writer(f).writerow(self.headers)
            return
        self._upgrade_csv_header()
        try:
            with open(path, "r") as f:
                lines = [line for line in f.read().splitlines() if line.strip()]
            if len(lines) > 1:
                last = lines[-1].split(",")
                agent.total_num_steps = int(float(last[0]))
                agent.total_num_updates = int(float(last[1]))
                print(f"{_DIM}    [setup] resuming at {agent.total_num_steps:,} env steps | "
                      f"{agent.total_num_updates:,} grad steps{_RST}")
        except Exception as e:
            print(f"[!] Warning: Could not recover steps from CSV: {e}")

    def _upgrade_csv_header(self):
        """Rewrites a CSV with other columns to the current header (matched by name; original kept as *.bak.csv)."""
        with open(self.csv_filename, "r", newline="") as f:
            rows = list(csv.DictReader(f))
            old = list(rows[0].keys()) if rows else None
        headers = self.headers
        if old is None or old == headers:
            return
        self._move_audit_columns(rows, old)
        backup = os.path.splitext(self.csv_filename)[0] + ".bak.csv"
        if not os.path.exists(backup):
            shutil.copy2(self.csv_filename, backup)
        with open(self.csv_filename, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore", restval="")
            w.writeheader()
            w.writerows(rows)
        print(f"[*] Rewrote {self.csv_filename} to the current {len(headers)} columns (original kept as {backup}).")

    @property
    def audit_headers(self):
        """Header row of the dream audit CSV."""
        return self.BASE_HEADERS[:2] + [c[0] for c in AUDIT_COLUMNS]

    def _append_audit_row(self, audit):
        """Appends one dream audit to its own CSV (created with its header when missing)."""
        path, agent = self.audit_filename, self.agent
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        new = not os.path.exists(path)
        row = [agent.total_num_steps, agent.total_num_updates] + [audit.get(key, "") for _, _, key in AUDIT_COLUMNS]
        row = [round(x, 4) if isinstance(x, float) else x for x in row]
        with open(path, mode="a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(self.audit_headers)
            w.writerow(row)

    def _move_audit_columns(self, rows, old_headers):
        """Moves audit columns found in the metrics CSV into the audit CSV (skipped when the audit CSV exists)."""
        names = [c[0] for c in AUDIT_COLUMNS]
        if os.path.exists(self.audit_filename) or not any(n in old_headers for n in names):
            return
        keep = [r for r in rows if any((r.get(n) or "").strip() for n in names)]
        if not keep:
            return
        os.makedirs(os.path.dirname(self.audit_filename) or ".", exist_ok=True)
        with open(self.audit_filename, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.audit_headers, extrasaction="ignore", restval="")
            w.writeheader()
            w.writerows(keep)
        print(f"[*] Moved the audit columns of {len(keep)} rounds to {self.audit_filename}.")

    def _append_csv_row(self, avg_score, metrics):
        """Appends one round of metrics to the CSV."""
        agent = self.agent
        row = [agent.total_num_steps, agent.total_num_updates, avg_score]
        row += [metrics[src].get(key, "") for _, src, key in self.columns]
        row = [round(x, 4) if isinstance(x, float) else x for x in row]
        with open(self.csv_filename, mode="a", newline="") as f:
            csv.writer(f).writerow(row)

    def _update_plots(self):
        """Refreshes the reward PNGs; plotting failures never stop training. The
        comparison draws every other run's CSV in metrics/ (find_comparison_runs), re-listed at each refresh."""
        try:
            runs = find_comparison_runs(self.csv_filename, self.reward_compare_labels, x_column="gradientSteps")
        except Exception as e:
            runs = []
            print(f"[!] Warning: could not list the earlier runs' CSVs: {e}")
        for name, fn in (("reward graph", lambda: plot_training(self.csv_filename, self.reward_plot_filename,
                                                                self.reward_ma_window)),
                         ("comparison graph (gradient updates)", lambda: plot_reward_comparison(
                             self.csv_filename, runs, self.reward_compare_filename, self.reward_ma_window,
                             x_column="gradientSteps"))):
            try:
                path = fn()
                if path is not None:
                    print(f"{_DIM}    [plot] {name} -> {path}{_RST}")
            except Exception as e:
                print(f"[!] Warning: could not update {name}: {e}")

    # --- Console output ---------------------------------------------------------------------------------------------
    def _print_round_header(self, rnd):
        """Banner with the round number and step counters."""
        bar = "═" * 62
        print(f"\n{_B}{_CYA}{bar}{_RST}")
        print(f"{_B}{_CYA}  ROUND {rnd + 1} / {self.total_rounds}{_RST}{_DIM}   env steps "
              f"{self.agent.total_num_steps:,} | grad steps {self.agent.total_num_updates:,}{_RST}")
        print(f"{_B}{_CYA}{bar}{_RST}")

    def _print_summary(self, avg_score, avg_curiosity, wm, dream, play, timing=None, mem=None, audit=None):
        """Round summary: reward, curiosity and progress up front, model diagnostics dimmed below."""
        agent, buf = self.agent, self.agent.buffer
        k = agent.cfg.rollout_steps
        lvl = play.get("furthest_level", -1)
        where = f"World {MAIN_LEVELS[lvl]}  x {play.get('furthest_x', 0)}" if 0 <= lvl < len(MAIN_LEVELS) else "–"
        print()
        print(f"  {_B}{_GRN}★ REWARD     {avg_score:8.2f}{_RST}{_GRN}   best {play.get('best_reward', 0):7.2f}{_RST}"
              f"      {_B}{_MAG}◆ CURIOSITY  {avg_curiosity:8.2f}{_RST}{_MAG}   best "
              f"{play.get('best_curiosity', 0):7.2f}{_RST}")
        print(f"  {_B}{_YEL}➜ FURTHEST   {where}{_RST}{_YEL}   (mean x {play.get('mean_x', 0):.0f}){_RST}"
              f"{_DIM}      avg episode {play.get('episode_length', 0):.0f} steps{_RST}")

        def row(label, text):
            print(f"{_DIM}    {label:<12}│ {text}{_RST}")

        def f(v, d=3):
            return "–" if v is None or v != v else f"{v:.{d}f}"

        print()
        if wm:                                                         # world model KL: nats and its loss terms
            kl_terms = " + ".join(f"{t} {wm[f'wm_{t}']:.3g}" for t in ("kl_dyn", "kl_rep") if f"wm_{t}" in wm)
            row("KL", f"{f(wm.get('kl_dyn'), 2)} nats" + (f" · loss {kl_terms}" if kl_terms else ""))
        if dream:
            row("actor", f"entropy {f(dream.get('entropies'))}")
        if timing:
            ph = timing.get("phases", {})
            enc = (f" (last update: posteriors {ph['encode']:.2f} s, decoder {ph.get('decode', 0.0):.2f} s, "
                   f"backward {ph.get('WM backward + step', 0.0):.2f} s, dreams "
                   f"{ph.get('dream + actor/critic', 0.0):.2f} s)") if "encode" in ph else ""
            row("speed", f"{timing['per_update']:.2f} s/update{enc} · play {play.get('t_play', 0):.0f} s")
        if not self.FULL_SUMMARY:
            for line in audit_lines(audit or {}):
                print(f"{_DIM}    {line}{_RST}")
            return
        if wm:
            row("encoder", f"recon excess {f(wm.get('recon_excess'), 4)} · sprite tiles "
                           f"{f(wm.get('recon_sprite_excess'), 4)} · posterior entropy {f(wm.get('post_entropy'), 1)} "
                           f"nats · classes used {f(wm.get('latent_usage'))} · drift {f(wm.get('encoder_drift'))}")
            row("grounding", f"z: level acc {f(wm.get('level_acc'))} (no token "
                             f"{f(wm.get('level_acc_notoken'))}) · tier acc {f(wm.get('tier_acc'))} · dreams: level "
                             f"acc {f(wm.get('dream_level_acc'))}, tier acc {f(wm.get('dream_tier_acc'))} · "
                             f"masked {f(wm.get('mask_ratio'))}")
            ro = (f"rollout KL {f(wm.get('rollout_loss'), 2)} (step {k}: {f(wm.get('rollout_kl_k'), 2)})" if k > 0
                  else "rollout off")
            row("dynamics", f"KL {f(wm.get('kl_dyn'), 2)} nats · prior entropy {f(wm.get('prior_entropy'), 1)} · "
                            f"{ro}")
            row("prediction", f"prior = posterior class {f(wm.get('pred_match'))} vs copy {f(wm.get('copy_match'))}"
                              + (f" · after {k} open-loop steps {f(wm.get('rollout_match_k'))}" if k > 0 else ""))
            row("heads", f"p(alive) at death {f(wm.get('p_alive_at_death'))} · elsewhere "
                         f"{f(wm.get('p_alive_alive'))} · reward {f(wm.get('reward_loss'))} · curiosity "
                         f"{f(wm.get('curiosity_loss'))} · [h, z] grid acc {f(wm.get('grid_acc'))} (Mario's tile "
                         f"{f(wm.get('grid_center_acc'))})")
            row("dream death", f"p(alive) at real deaths · 1 dream step {f(wm.get('p_alive_death_pred1'))} · after "
                               f"{agent.DEATH_PROBE_STEPS} open-loop steps {f(wm.get('p_alive_death_dream'))} · "
                               f"survival {f(wm.get('dream_death_survival'))} · "
                               f"{wm.get('death_probe_deaths', 0):.0f} deaths")
            groups = ("pixel", "grounding", "kl", "rollout", "reward", "cont", "grid")
            if "harmony_pixel" in wm:
                row("harmony", "loss weights 1/σ · " + " · ".join(
                    f"{g} {wm[f'harmony_{g}']:.3g}" for g in groups if f"harmony_{g}" in wm))
            total = wm.get("world_model_loss")
            if total:                                                  # weighted terms, round means, share of total
                row("loss terms", f"total {total:.4g} = " + " + ".join(
                    f"{t} {wm[f'wm_{t}']:.3g} ({100.0 * wm[f'wm_{t}'] / total:.1f}%)"
                    for t in agent.WM_LOSS_TERMS if f"wm_{t}" in wm))
            row("gradients", f"norm {f(wm.get('wm_grad_norm'), 1)} · clip at {f(wm.get('wm_clip_threshold'), 1)}"
                             f" · clipped {100.0 * wm.get('wm_clip_fraction', 0.0):.1f}% of updates")
            for g, label in (("tssm", "TSSM"), ("encoder", "encoder")):
                shares = [(t, wm[f"gshare_{g}_{t}"]) for t in agent.WM_LOSS_TERMS if f"gshare_{g}_{t}" in wm]
                shares = [(t, v) for t, v in shares if v >= 0.005]
                if shares:
                    row("grad shares" if g == "tssm" else "", f"{label} (G{wm.get('grad_share_update', 0):.0f}): "
                        + " · ".join(f"{t} {100.0 * v:.0f}%" for t, v in sorted(shares, key=lambda x: -x[1])))
        if dream:
            row("actor/critic", f"actor loss {f(dream.get('actor_loss'), 4)} · critic loss "
                                f"{f(dream.get('critic_loss'), 4)} · advantage reward {f(dream.get('advantages'), 4)}"
                                f" / curiosity {f(dream.get('curiosity_advantages'), 4)}")
            row("critics", f"fit (explained variance) reward {f(dream.get('ext_critic_ev'), 2)} / curiosity "
                           f"{f(dream.get('cur_critic_ev'), 2)} · share of the actor's advantage reward "
                           f"{100.0 * dream.get('ext_adv_share', float('nan')):.0f}% / curiosity "
                           f"{100.0 * dream.get('cur_adv_share', float('nan')):.0f}% · combined std "
                           f"{f(dream.get('adv_combined_std'), 3)} · dream death penalty "
                           f"{f(dream.get('dream_death_penalty'), 4)} / step")
            row("behaviour", f"entropy {f(dream['entropies'])} · dream p(alive) {f(dream['dream_mean_alive'])} · "
                             f"dream curiosity {f(dream['dream_mean_curiosity'], 4)} · curiosity flips "
                             f"{100.0 * dream['curiosity_flip_fraction']:.1f}% · critic "
                             f"{f(dream['critic_values'])} (real {f(dream['replay_returns'])})")
            scales = (f"advantage std reward {f(dream.get('adv_scale'), 3)} / curiosity "
                      f"{f(dream.get('cur_adv_scale'), 3)}"
                      + f" · positive {100.0 * dream.get('advantage_pos_fraction', 0.0):.0f}% · mean |A| "
                      f"{f(dream.get('adv_abs_mean'), 4)}")
            row("REINFORCE", scales)
            if "branch_agree" in dream or dream.get("dream_start_pre_death", 0.0) > 0:
                txt = (f"pre-death {100.0 * dream.get('dream_start_pre_death', 0.0):.0f}% of dreams "
                       f"({f(wm.get('pre_death_states'), 0)} pre-death states in the batch)")
                if "branch_agree" in dream:
                    txt += (f" · branches: the policy's favourite first action is the best one in "
                            f"{100.0 * dream['branch_agree']:.0f}% of groups, "
                            f"spread {f(dream.get('branch_spread'), 3)}")
                row("dream starts", txt)
        if mem:
            row("gpu memory", f"peak {mem['gpu_peak_gb']:.1f} GB · reserved {mem['gpu_reserved_gb']:.1f} GB · "
                              f"allocated {mem['gpu_allocated_gb']:.1f} GB · free on device {mem['gpu_free_gb']:.1f} / "
                              f"{mem['gpu_total_gb']:.1f} GB")
        row("buffer", f"{100.0 * len(buf) / buf.capacity:.0f}% full · {len(buf):,} steps · non-playable steps "
                      f"{100.0 * play.get('nonplayable_steps', 0.0):.1f}% · skipped {play.get('skipped_steps', 0)} · "
                      f"death penalty {play.get('death_penalty', 0.0):.2f} / episode")
        for line in audit_lines(audit or {}):
            print(f"{_DIM}    {line}{_RST}")

    # --- Evaluation -------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, num_episodes=None, recording=False, greedy=False, checkpoint=None, max_steps=None):
        """Every env plays `num_episodes` (None: num_eval_episodes of the config) with the policy of `checkpoint`
        (None: the newest one in checkpoints/); recording=True writes one MP4 per env; max_steps ends an episode
        early."""
        if num_episodes is None:
            num_episodes = self.num_eval_episodes
        agent = self.agent
        if checkpoint is not None:
            if not os.path.exists(checkpoint):
                raise FileNotFoundError(f"no checkpoint at {checkpoint}")
            if os.path.getsize(checkpoint) < 1_000_000:
                raise RuntimeError(f"{checkpoint} is a Git LFS pointer, not the model: run `git lfs pull` first")
        agent.load_checkpoint(path=checkpoint, load_buffer=False)
        print(f"Starting Parallel Evaluation for {num_episodes} episodes per agent...")
        num_envs = len(self.envs)

        writers, frame_size = [], (480, 432)                  # 3x the full 160x144 screen (HUD included)
        if recording:
            import cv2
            os.makedirs("recordings", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.envs.set_capture(True)                       # every emulator frame, not one per agent step
            writers = [cv2.VideoWriter(os.path.join("recordings", f"eval_agent{i + 1}.mp4"), fourcc, RECORD_FPS,
                                       frame_size) for i in range(num_envs)]
            print(f"{_DIM}    [rec] recording {num_envs} agents at {RECORD_FPS} fps -> "
                  f"recordings/eval_agent1..{num_envs}.mp4{_RST}")

        def record(i, info):
            """Writes the emulator frames of env i's last step / reset to its video."""
            if writers:
                import cv2
                for frame in info["frames"]:
                    frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_NEAREST)
                    writers[i].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        done_eps = [0] * num_envs
        rewards = [0.0] * num_envs
        steps = [0] * num_envs
        ctx = agent.new_act_context(num_envs)
        try:
            observations = []
            for i, (obs, info) in enumerate(self.envs.reset()):
                observations.append(obs)
                record(i, info)

            while min(done_eps) < num_episodes:
                action_idxs = torch.argmax(agent.act(observations, ctx, greedy=greedy), dim=-1).cpu().numpy()
                active = [done_eps[i] < num_episodes for i in range(num_envs)]
                results = self.envs.step(action_idxs, active=active)
                to_reset = []
                for i in range(num_envs):
                    if not active[i]:
                        continue
                    obs, reward, terminated, truncated, info = results[i]
                    observations[i] = obs
                    record(i, info)
                    rewards[i] += reward
                    steps[i] += 1
                    if steps[i] % 500 == 0:
                        print(f"[Agent {i+1}] Eval Step: {steps[i]}/{info['limit']} | Reward: {rewards[i]:.3f}")
                    if terminated or truncated or (max_steps and steps[i] >= max_steps):
                        how = "died" if info.get("died") else "time-out" if info.get("timeout") else \
                            ("terminated" if terminated else "budget")
                        print(f"--- [Agent {i+1}] Episode Finished ({how})! Reward: {rewards[i]:.3f} | "
                              f"Steps: {steps[i]} ---")
                        done_eps[i] += 1
                        if done_eps[i] < num_episodes:
                            to_reset.append(i)
                            rewards[i], steps[i] = 0.0, 0
                            agent.reset_act_rows(ctx, i)
                for i in to_reset:
                    observations[i], info = self.envs.reset_one(i)
                    record(i, info)
        finally:                                              # also on Ctrl+C: an unreleased MP4 does not play
            for i, w in enumerate(writers):
                w.release()
                print(f"{_DIM}    [rec] saved recordings/eval_agent{i + 1}.mp4{_RST}")
            if writers:
                try:
                    self.envs.set_capture(False)
                except Exception:
                    pass


# --- Notebook entry point -------------------------------------------------------------------------------------------
def run(mode="train", num_envs=25, verbose=False, recording=False, visualize_dreams=True,
        rom_path="game/SuperMarioBrosDeluxe.gbc", reset_actor_critic=False, device=None, checkpoint=None,
        window=True, max_steps=None, **overrides):
    """Creates the envs, trains or evaluates GQ-Dreamer and closes the envs -> the Trainer.

    mode: "train" (num_envs headless emulators) or "eval" (num_eval_envs SDL2 windows, num_eval_episodes each;
    both set in GQDreamerConfig, overridable here as run(..., num_eval_envs=5));
    verbose: per-env agent logs; recording: eval writes one MP4 per env to recordings/; visualize_dreams: dream PDFs
    each training round in dreams/; reset_actor_critic: keep world model + buffer, restart actor / critics;
    checkpoint: eval only, the model file to play (None: the newest in checkpoints/); window: eval in SDL2 windows
    (False: headless); max_steps: eval episodes end after this many steps; overrides: GQDreamerConfig fields. Ctrl+C / "interrupt kernel" stops cleanly and still returns the Trainer."""
    if mode not in ("train", "eval"):
        raise ValueError(f"invalid mode {mode!r}: use 'train' or 'eval'")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    if mode == "eval":
        n, speed = model_config(**overrides).num_eval_envs, int(window)
        window = "SDL2" if window else "null"
        overrides.setdefault("buffer_size", 1000)                      # eval needs no 10 GB replay buffer
    else:
        n, window, speed = num_envs, "null", 0
    envs = create_envs(num_envs=n, rom_path=rom_path, state_dir="game", wind=window, speed=speed,
                       skip_transitions=model_config(**overrides).skip_transitions)
    trainer = None
    try:
        print("Initializing GQ-Dreamer...")
        trainer = Trainer(envs, device=device, visualize_dreams=visualize_dreams, verbose=verbose, **overrides)
        if mode == "train":
            print("Starting Training Mode...")
            trainer.train(reset_actor_critic=reset_actor_critic)
        else:
            print("Starting Evaluation Mode...")
            trainer.evaluate(recording=recording, checkpoint=checkpoint, max_steps=max_steps)
    except KeyboardInterrupt:
        print("\n[!] Interrupted by the user.")
    finally:
        envs.close()
        print("All environments safely closed.")
    return trainer
