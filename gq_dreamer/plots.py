"""Dark-theme plots of the metrics CSV: reward per round with the furthest level reached, and this run's reward
against other runs."""
import csv
import glob
import os
import time

import numpy as np

# --- Dark theme -----------------------------------------------------------------------------------------------------
SURFACE = "#1a1a19"          # figure + plot background
INK = "#ffffff"              # titles, the end value
INK_2 = "#c3c2b7"            # subtitles, level labels, legend
MUTED = "#898781"            # axis ticks and labels
GRID = "#2c2c2a"             # hairline gridlines
BASELINE = "#383835"         # zero line, level boundaries
BAND = "#ffffff"             # alternate level shading (at a few % opacity)
BLUE = "#3987e5"             # this run
ORANGE = "#d95926"           # the previous run
TEAL = "#2a9d8f"             # other earlier runs
GREEN = "#8ab17d"
YELLOW = "#e2b33c"
PURPLE = "#9d7bd8"
PINK = "#d0679d"
EARLIER_COLOURS = (TEAL, GREEN, YELLOW, PURPLE, PINK)

_STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": MUTED, "text.color": INK_2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.labelsize": 10.5, "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,
}


# --- Data -----------------------------------------------------------------------------------------------------------
def moving_average(values, window):
    """Trailing moving average over the last `window` records (NaNs ignored)."""
    v = np.asarray(values, dtype=np.float64)
    out = np.full_like(v, np.nan)
    for i in range(len(v)):
        seg = v[max(0, i - int(window) + 1):i + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = seg.mean()
    return out


def moving_std(values, window):
    """Trailing standard deviation over the last `window` records (NaNs ignored)."""
    v = np.asarray(values, dtype=np.float64)
    out = np.full_like(v, np.nan)
    for i in range(len(v)):
        seg = v[max(0, i - int(window) + 1):i + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = seg.std()
    return out


def read_reward_csv(path, x_column):
    """(x, totalReward) arrays of a metrics CSV, skipping unreadable rows."""
    xs, rewards = [], []
    with open(path, mode="r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                x, reward = float(row[x_column]), float(row["totalReward"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(x) and np.isfinite(reward):
                xs.append(x)
                rewards.append(reward)
    return np.asarray(xs, dtype=np.float64), np.asarray(rewards, dtype=np.float64)


def _has_reward_rows(path, x_column):
    """True when the CSV has x_column and totalReward columns and at least one row with both readable."""
    try:
        with open(path, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            if not {x_column, "totalReward"} <= set(reader.fieldnames or ()):
                return False
            for row in reader:
                try:
                    if np.isfinite(float(row[x_column])) and np.isfinite(float(row["totalReward"])):
                        return True
                except (TypeError, ValueError):
                    continue
    except (OSError, UnicodeDecodeError, csv.Error):
        return False
    return False


def _archive_label(path, stem):
    """'run archived <date> <time>' from <stem>_before_<YYYYmmdd_HHMMSS>.csv (the file name if it has no date)."""
    tag = os.path.splitext(os.path.basename(path))[0][len(f"{stem}_before_"):]
    try:
        return "run archived " + time.strftime("%d %b %H:%M", time.strptime(tag, "%Y%m%d_%H%M%S"))
    except ValueError:
        return f"run before {tag}"


def find_comparison_runs(csv_path, labels=None, x_column="envSteps", max_runs=6):
    """Earlier runs [(label, [path], colour)] for plot_reward_comparison: every other CSV next to csv_path with rows
    of x_column and totalReward (the trainer's training_data CSVs, or any run exported in that format).
    The newest <stem>_before_<date>.csv, archived when a fresh model starts, is the "previous run" (orange); labels
    {file name: label} names the rest, else the file name without its "<stem>_" / "compare_" prefix
    (training_data_v1.csv -> "v1"). Then the other files (alphabetical, so their colours stay put), then the older
    archives (newest first), at most max_runs in all."""
    folder = os.path.dirname(csv_path) or "."
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    labels = labels or {}
    current = os.path.abspath(csv_path)
    files = sorted(p for p in glob.glob(os.path.join(folder, "*.csv"))
                   if os.path.abspath(p) != current and _has_reward_rows(p, x_column))
    archived = sorted((p for p in files if os.path.basename(p).startswith(f"{stem}_before_")), reverse=True)
    others = [p for p in files if p not in archived]

    def name(p):
        base = os.path.basename(p)
        if base in labels:
            return labels[base]
        tag = os.path.splitext(base)[0]
        for prefix in (f"{stem}_", "compare_"):
            if tag.startswith(prefix) and len(tag) > len(prefix):
                tag = tag[len(prefix):]
                break
        return tag.replace("_", " ")

    runs = [(labels.get(os.path.basename(archived[0]), "previous run"), [archived[0]], ORANGE)] if archived else []
    rest = [(name(p), p) for p in others]
    rest += [(labels.get(os.path.basename(p)) or _archive_label(p, stem), p) for p in archived[1:]]
    runs += [(label, [p], EARLIER_COLOURS[i % len(EARLIER_COLOURS)]) for i, (label, p) in enumerate(rest)]
    return runs[:max_runs]


def _read(path):
    """All rows of a CSV as dicts."""
    with open(path, mode="r", newline="") as f:
        return list(csv.DictReader(f))


def _col(rows, name):
    """Column `name` of the rows as a float array (NaN where missing or unreadable)."""
    out = []
    for r in rows:
        try:
            out.append(float(r.get(name, "nan")))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out)


def _save_figure(fig, path, dpi=130):
    """Saves and closes the figure through a temporary file, so a viewer never sees a half-written PNG."""
    import matplotlib.pyplot as plt
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, f".{os.path.basename(path)}.tmp.png")
    try:
        fig.savefig(tmp, dpi=dpi, facecolor=fig.get_facecolor())
        os.replace(tmp, path)
    finally:
        plt.close(fig)
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


# --- Shared pieces --------------------------------------------------------------------------------------------------
def _compact(v, _pos=None):
    """An axis number as 0 / 250K / 1.5M."""
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.2f}".rstrip("0").rstrip(".") + "M"
    if a >= 1e3:
        return f"{v / 1e3:.1f}".rstrip("0").rstrip(".") + "K"
    return f"{v:g}"


def _level_label(index):
    """Main-level index (game.MAIN_LEVELS) -> 'furthest 1-2'."""
    from .game import MAIN_LEVELS
    i = int(index)
    return f"furthest {MAIN_LEVELS[i]}" if 0 <= i < len(MAIN_LEVELS) else "no level yet"


def _style_axes(ax, x_label):
    """Horizontal hairline grid, bottom baseline only, muted ticks, compact x numbers."""
    import matplotlib.ticker as mticker
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(True, axis="y", color=GRID, lw=0.8, ls="-")
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", length=4, color=BASELINE, pad=6)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_compact))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:,.0f}"))
    ax.set_xlabel(x_label, labelpad=8)


def _header(fig, ax, title, subtitle):
    """Title and subtitle aligned with the plot's left edge."""
    left = ax.get_position().x0
    fig.text(left, 0.955, title, color=INK, fontsize=17, fontweight="semibold", ha="left", va="top")
    fig.text(left, 0.905, subtitle, color=INK_2, fontsize=10.5, ha="left", va="top")


def _end_marker(ax, x, y, color, caption):
    """A haloed dot on the last point of a curve with its value and a caption to the right."""
    ax.scatter([x], [y], s=380, color=color, alpha=0.16, lw=0, zorder=5)
    ax.scatter([x], [y], s=70, color=color, edgecolors=SURFACE, linewidths=2.0, zorder=6)
    ax.annotate(f"{y:,.1f}", (x, y), xytext=(14, 5), textcoords="offset points", ha="left", va="center",
                color=INK, fontsize=14, fontweight="semibold", zorder=7)
    ax.annotate(caption, (x, y), xytext=(14, -11), textcoords="offset points", ha="left", va="center",
                color=MUTED, fontsize=8.5, zorder=7)


def _y_limits(ax, *series):
    """The y range of the finite values of the given arrays, padded."""
    vals = np.concatenate([np.asarray(s, dtype=np.float64).ravel() for s in series])
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        return
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.08 * max(hi - lo, 1.0)
    ax.set_ylim(lo - pad, hi + pad)


# --- Figures --------------------------------------------------------------------------------------------------------
def plot_training(csv_path, out_path, window=20, title="GQ-Dreamer · reward"):
    """Reward per round over env steps: raw (thin), moving average +- 1 std (line and band), the latest average marked
    at the end, and the furthest main level reached so far shaded and labelled (column furthestLevel). Returns the
    path, or None if the CSV has no rows."""
    if not os.path.exists(csv_path):
        return None
    rows = _read(csv_path)
    if not rows:
        return None
    import matplotlib.pyplot as plt
    x = _col(rows, "envSteps")
    reward = _col(rows, "totalReward")
    keep = np.isfinite(x)
    rows = [r for r, k in zip(rows, keep) if k]
    x, reward = x[keep], reward[keep]
    if not len(x):
        return None
    grad = _col(rows, "gradientSteps")
    level = _col(rows, "furthestLevel")
    ma = moving_average(reward, window)
    sd = moving_std(reward, window)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(14, 7.2))
        fig.subplots_adjust(left=0.065, right=0.975, top=0.84, bottom=0.1)
        span_x = max(float(x[-1] - x[0]), 1.0)

        # furthest level reached so far: alternate faint shading, a hairline where it advances, its name at the top
        if np.isfinite(level).any():
            best = np.fmax.accumulate(np.where(np.isfinite(level), level, -1.0))
            spans, start = [], 0
            for i in range(1, len(best) + 1):
                if i == len(best) or best[i] != best[start]:
                    spans.append((x[start], x[i] if i < len(best) else x[-1], best[start]))
                    start = i
            for k, (a, b, lvl) in enumerate(spans):
                if k % 2:
                    ax.axvspan(a, b, color=BAND, alpha=0.025, lw=0, zorder=0)
                if k > 0:
                    ax.axvline(a, color=BASELINE, lw=1.0, zorder=1)
                if (b - a) / span_x > 0.06 or len(spans) == 1:             # skip labels of very short spans
                    ax.text(a, 0.985, f"  {_level_label(lvl)}", transform=ax.get_xaxis_transform(), ha="left",
                            va="top", color=INK_2, fontsize=9.5, zorder=7)

        ax.fill_between(x, ma - sd, ma + sd, color=BLUE, alpha=0.12, lw=0, zorder=2)
        ax.plot(x, reward, color=BLUE, lw=1.0, alpha=0.35, zorder=2)
        ax.plot(x, ma, color=BLUE, lw=2.4, solid_capstyle="round", solid_joinstyle="round", zorder=3)

        _y_limits(ax, reward, ma - sd, ma + sd)
        lo, hi = ax.get_ylim()
        if lo < 0 < hi:
            ax.axhline(0, color=BASELINE, lw=1.2, zorder=1)
        ax.set_xlim(x[0] - 0.01 * span_x, x[-1] + 0.11 * span_x)          # room for the end label

        last = np.flatnonzero(np.isfinite(ma))
        if last.size:
            i = int(last[-1])
            _end_marker(ax, x[i], ma[i], BLUE, f"{window}-round avg")

        _style_axes(ax, "Environment steps")
        ax.set_ylabel("Reward per round", labelpad=10)
        g = grad[np.isfinite(grad)]
        steps = f"{_compact(x[-1])} env steps" + (f" · {int(g[-1]):,} gradient steps" if g.size else "")
        best_ma = np.nanmax(ma) if np.isfinite(ma).any() else np.nan
        best_txt = f" · best average {best_ma:,.1f}" if np.isfinite(best_ma) else ""
        _header(fig, ax, title,
                f"Total reward per round (thin), {window}-round moving average (line) ± 1 std (band) · {steps}"
                f"{best_txt}")
        return _save_figure(fig, out_path)


def plot_reward_comparison(csv_path, runs, out_path, window=20, current_label="GQ-Dreamer (current run)",
                           x_column="envSteps"):
    """This run's reward against earlier runs [(label, candidate CSV paths, colour)] over x_column ("envSteps" or
    "gradientSteps"): raw rows thin, moving averages thick, zoomed on this run's range, its latest average marked.
    Returns the path, or None without an earlier run that has rows (find_comparison_runs lists them)."""
    found = []
    for name, paths, colour in runs:
        path = next((p for p in paths if p and os.path.exists(p)), None)
        if path is not None and os.path.abspath(path) != os.path.abspath(csv_path):
            found.append((name, path, colour))
    data = [(name, *read_reward_csv(path, x_column), colour) for name, path, colour in found]
    data = [d for d in data if len(d[2])]                             # earlier runs with rows
    if not data:
        return None
    cur = read_reward_csv(csv_path, x_column) if os.path.exists(csv_path) else (np.zeros(0), np.zeros(0))
    data.append((current_label, *cur, BLUE))

    import matplotlib.pyplot as plt
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(14, 7.2))
        fig.subplots_adjust(left=0.065, right=0.975, top=0.84, bottom=0.1)
        for name, steps, rewards, colour in data:                     # raw curves below, averages on top
            if len(rewards):
                ax.plot(steps, rewards, color=colour, lw=1.0, alpha=0.3, zorder=2)
        for name, steps, rewards, colour in data:
            if len(rewards):
                ax.plot(steps, moving_average(rewards, window), color=colour, lw=2.4, solid_capstyle="round",
                        label=name, zorder=3)
        _, cur_x, cur_r, cur_c = data[-1]
        if len(cur_x) and cur_x.max() > 0:                            # zoom on this run's range
            x_max = cur_x.max() * 1.15
            ax.set_xlim(0, x_max)
            vis = [r[st <= x_max] for _, st, r, _ in data if len(r)]
            _y_limits(ax, *vis)
        lo, hi = ax.get_ylim()
        if lo < 0 < hi:
            ax.axhline(0, color=BASELINE, lw=1.2, zorder=1)
        if len(cur_r):
            ma = moving_average(cur_r, window)
            _end_marker(ax, cur_x[-1], ma[-1], cur_c, f"{window}-round avg")
        _style_axes(ax, "Gradient steps" if x_column == "gradientSteps" else "Environment steps")
        ax.set_ylabel("Reward per round", labelpad=10)
        leg = ax.legend(loc="upper left", frameon=False, fontsize=10, handlelength=1.6)
        for t in leg.get_texts():
            t.set_color(INK_2)
        _header(fig, ax, f"{current_label.split(' (')[0]} · reward vs earlier runs",
                f"Total reward per round (thin) and {window}-round moving average (line), over this run's range")
        return _save_figure(fig, out_path)
