from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from config import FIGURES_DIR, METHOD_COLORS, METHOD_LABELS, METHODS, MODELS_DIR, PAPER_FIGURES_DIR


DATASETS = ("edgeiiot", "ciciot")
DATASET_LABELS = {"edgeiiot": "Edge-IIoTset", "ciciot": "CICIoT2023"}
POSITIVE_CMAP = LinearSegmentedColormap.from_list("positive", ["#FFFFFF", "#FF4F4F"])
SIGNED_CMAP = LinearSegmentedColormap.from_list("signed", ["#007FFF", "#FFFFFF", "#FF4F4F"])

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 8.2,
    "axes.labelsize": 8.4,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 8.4,
    "axes.linewidth": 0.75,
    "lines.linewidth": 1.45,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.025,
})


def style_axis(ax: plt.Axes, grid: bool = True) -> None:
    ax.tick_params(direction="in", width=0.7, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    if grid:
        ax.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.65, zorder=0)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.12, label, transform=ax.transAxes, fontsize=9.2,
            fontweight="bold", va="bottom", ha="left", clip_on=False)


def angle_method_ticks(ax: plt.Axes, axis: str) -> None:
    """Angle categorical method ticks to preserve dense multi-method layouts."""
    labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    rotation = 42 if axis == "x" else 28
    for label in labels:
        label.set_rotation(rotation)
        label.set_ha("right")
        label.set_va("top" if axis == "x" else "center")


def load_json_runs() -> List[dict]:
    output = []
    for path in sorted(MODELS_DIR.glob("*.json")):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "final" in run:
            output.append(run)
    return output


def is_main(run: dict) -> bool:
    c = run["config"]
    return (
        c.get("variant", "full") == "full" and c["dirichlet_alpha"] == 0.3
        and c["max_staleness"] == 2 and c["participation"] == 0.6
        and c["seed"] in range(5)
    )


def select(runs: List[dict], **conditions: object) -> List[dict]:
    selected = []
    for run in runs:
        cfg = run["config"]
        if all(cfg.get(key) == value for key, value in conditions.items()):
            selected.append(run)
    return selected


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, format="pdf", dpi=600)
    plt.close(fig)
    destination = PAPER_FIGURES_DIR / name
    if destination.resolve() != path.resolve():
        shutil.copy2(path, destination)
    print(f"[figure] {path}")


def aggregate_curves(runs: List[dict], metric: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    curves = np.asarray([[r[metric] for r in run["records"]] for run in runs], dtype=float)
    mean = curves.mean(axis=0)
    spread = curves.std(axis=0, ddof=1)
    return np.arange(1, curves.shape[1] + 1), mean, spread


def plot_main_dynamics(runs: List[dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 5.05), constrained_layout=False)
    fig.subplots_adjust(left=0.072, right=0.988, bottom=0.185, top=0.925, wspace=0.40, hspace=0.43)

    # (a) Descriptive mean +/- one SD across dataset-seed trajectories.
    ax = axes[0, 0]
    for method in METHODS:
        method_runs = [r for r in runs if is_main(r) and r["config"]["method"] == method]
        x, mean, ci = aggregate_curves(method_runs, "macro_f1")
        ax.plot(x, mean, color=METHOD_COLORS[method], label=METHOD_LABELS[method], zorder=3)
        ax.fill_between(x, mean - ci, mean + ci, color=METHOD_COLORS[method], alpha=0.10, linewidth=0)
    for boundary in (5, 10, 15, 20):
        ax.axvline(boundary, color="#888888", ls="--", lw=0.65, alpha=0.7)
    ax.set(xlabel="Communication round", ylabel="Macro-F1 score $\\uparrow$", xlim=(1, 25), ylim=(0.12, 0.82))
    ax.set_title("Continual performance (mean $\\pm$ 1 SD)", pad=3)
    style_axis(ax); panel_label(ax, "(a)")

    # (b) Calibration evolution.
    ax = axes[0, 1]
    for method in METHODS:
        method_runs = [r for r in runs if is_main(r) and r["config"]["method"] == method]
        x, mean, ci = aggregate_curves(method_runs, "ece")
        ax.plot(x, mean, color=METHOD_COLORS[method])
        ax.fill_between(x, np.maximum(0, mean - ci), mean + ci, color=METHOD_COLORS[method], alpha=0.10, linewidth=0)
    ax.axhline(0.10, color="#333333", ls=":", lw=0.8)
    ax.set(xlabel="Communication round", ylabel="Expected calibration error $\\downarrow$", xlim=(1, 25), ylim=(0, 0.42))
    ax.set_title("Calibration under concept exposure", pad=3)
    style_axis(ax); panel_label(ax, "(b)")

    # (c) Client-level distribution.
    ax = axes[0, 2]
    distributions = []
    for method in METHODS:
        values = []
        for run in runs:
            if is_main(run) and run["config"]["method"] == method:
                values.extend(run["final"]["client_macro_f1"])
        distributions.append(values)
    violin = ax.violinplot(distributions, showmedians=True, showextrema=False, widths=0.82)
    for body, method in zip(violin["bodies"], METHODS):
        body.set_facecolor(METHOD_COLORS[method]); body.set_edgecolor("#333333"); body.set_alpha(0.72); body.set_linewidth(0.55)
    violin["cmedians"].set_color("#111111"); violin["cmedians"].set_linewidth(1.0)
    ax.set_xticks(range(1, len(METHODS) + 1), [METHOD_LABELS[m] for m in METHODS])
    angle_method_ticks(ax, "x")
    ax.set(ylabel="Client macro-F1 score $\\uparrow$", ylim=(0, 1.02))
    ax.set_title("Cross-client fairness distribution", pad=3)
    style_axis(ax); panel_label(ax, "(c)")

    # (d) Accuracy-efficiency Pareto map.
    ax = axes[1, 0]
    for method in METHODS:
        selected = [r for r in runs if is_main(r) and r["config"]["method"] == method]
        f1 = np.mean([r["final"]["macro_f1"] for r in selected])
        comm = np.mean([r["final"]["communication_mb"] for r in selected])
        memory = np.mean([r["final"]["peak_auxiliary_memory_mb"] for r in selected])
        size = 32 + 800 * memory
        ax.scatter(comm, f1, s=size, color=METHOD_COLORS[method], edgecolor="#222222", linewidth=0.55, zorder=3)
    ours_runs = [r for r in runs if is_main(r) and r["config"]["method"] == "fedgeski"]
    ours_xy = (
        np.mean([r["final"]["communication_mb"] for r in ours_runs]),
        np.mean([r["final"]["macro_f1"] for r in ours_runs]),
    )
    ax.annotate("FedGeSKI", xy=ours_xy, xytext=(-58, -24), textcoords="offset points",
                ha="left", va="top", arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=7)
    ax.set(xlabel="Cumulative communication (MiB) $\\downarrow$", ylabel="Macro-F1 score $\\uparrow$")
    ax.set_title("Accuracy--resource Pareto map", pad=3)
    style_axis(ax); panel_label(ax, "(d)")

    # (e) Positive-valued class recall heat map.
    ax = axes[1, 1]
    matrix = []
    for method in METHODS:
        values = [np.asarray(r["final"]["per_class_recall"]) for r in runs if is_main(r) and r["config"]["method"] == method]
        matrix.append(np.mean(values, axis=0))
    matrix = np.asarray(matrix)
    im = ax.imshow(matrix, cmap=POSITIVE_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(6), [f"C{i}" for i in range(6)])
    ax.set_yticks(range(len(METHODS)), [METHOD_LABELS[m] for m in METHODS])
    angle_method_ticks(ax, "y")
    for i in range(len(METHODS)):
        for j in range(6):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=5.9, color="#111111")
    ax.set_title("Per-class recall $\\uparrow$", pad=3)
    ax.tick_params(direction="out", length=2); panel_label(ax, "(e)")

    # (f) Dataset-wise method ranks; lower is better.
    ax = axes[1, 2]
    metrics = [("macro_f1", False), ("balanced_accuracy", False), ("macro_auroc", False), ("ece", True), ("average_forgetting", True)]
    rank_rows = []
    for dataset in DATASETS:
        scores = {m: [] for m in METHODS}
        for metric, lower in metrics:
            vals = {}
            for method in METHODS:
                chosen = [r["final"][metric] for r in runs if is_main(r) and r["config"]["dataset"] == dataset and r["config"]["method"] == method]
                vals[method] = np.mean(chosen)
            ranked = sorted(METHODS, key=lambda m: vals[m], reverse=not lower)
            for idx, method in enumerate(ranked, 1): scores[method].append(idx)
        for method in METHODS: rank_rows.append((dataset, method, np.mean(scores[method])))
    for yi, method in enumerate(METHODS):
        vals = [row[2] for row in rank_rows if row[1] == method]
        ax.plot(vals, [yi - 0.10, yi + 0.10], color=METHOD_COLORS[method], lw=1.2)
        ax.scatter(vals, [yi - 0.10, yi + 0.10], color=METHOD_COLORS[method], s=28, edgecolor="#222222", linewidth=0.5, zorder=3)
    ax.set_yticks(range(len(METHODS)), [METHOD_LABELS[m] for m in METHODS])
    angle_method_ticks(ax, "y")
    ax.set_xticks(range(1, len(METHODS) + 1)); ax.invert_xaxis(); ax.invert_yaxis()
    ax.set(xlabel="Average rank $\\downarrow$ (1 = best)", xlim=(len(METHODS) + 0.2, 0.8))
    ax.set_title("Multi-metric ranking", pad=3)
    style_axis(ax); panel_label(ax, "(f)")

    compact = {"fedgcc":"Fed-GCC", "afcl_csc":"AFCL-CSC", "fedavgm":"FAvgM",
               "fedadam":"FAdam", "fedyogi":"FYogi", "fedasync":"FAsync",
               "fedbuff":"FBuff", "glfc":"GLFC", "evofedids":"EvoIDS",
               "fedta":"FTA", "fedagc":"FAGC", "fedgeski":"FedGeSKI"}
    line_handles = [Line2D([0], [0], color=METHOD_COLORS[m], marker="o", markersize=3.4,
                           lw=1.8, label=compact[m]) for m in METHODS]
    circle_handles = [Line2D([0], [0], marker="o", linestyle="None", markersize=5.2,
                             markerfacecolor=METHOD_COLORS[m], markeredgecolor="#222222",
                             markeredgewidth=0.45, label=compact[m]) for m in METHODS]
    fig.legend(handles=line_handles, loc="lower center", ncol=12, frameon=False,
               bbox_to_anchor=(0.5, 0.055), fontsize=7.5, columnspacing=0.30,
               handletextpad=0.20, handlelength=1.05)
    fig.legend(handles=circle_handles, loc="lower center", ncol=12, frameon=False,
               bbox_to_anchor=(0.5, 0.012), fontsize=7.5, columnspacing=0.30,
               handletextpad=0.22, handlelength=0.75)
    save_figure(fig, "main_dynamics.pdf")


def stress_value(runs: List[dict], dataset: str, method: str, key: str, value: float) -> float:
    conditions = dict(dataset=dataset, method=method, variant="full", seed=0)
    conditions.update({key: value})
    if key != "dirichlet_alpha": conditions["dirichlet_alpha"] = 0.3
    if key != "max_staleness": conditions["max_staleness"] = 2
    if key != "participation": conditions["participation"] = 0.6
    chosen = select(runs, **conditions)
    return float(chosen[0]["final"]["macro_f1"]) if chosen else np.nan


def plot_robustness(runs: List[dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 5.05))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.185, top=0.925, wspace=0.39, hspace=0.44)
    # (a) staleness
    ax = axes[0, 0]; xs = [0, 1, 2, 3, 4]
    for method in METHODS:
        ys = [np.mean([stress_value(runs, d, method, "max_staleness", x) for d in DATASETS]) for x in xs]
        ax.plot(xs, ys, marker="o", ms=3.2, color=METHOD_COLORS[method])
    ax.set(xlabel="Maximum model age (rounds)", ylabel="Macro-F1 score $\\uparrow$", xticks=xs)
    ax.set_title("Bounded-age tolerance", pad=3); style_axis(ax); panel_label(ax, "(a)")

    # (b) Dirichlet heterogeneity
    ax = axes[0, 1]; xs = [0.1, 0.3, 0.6, 1.0]
    for method in METHODS:
        ys = [np.mean([stress_value(runs, d, method, "dirichlet_alpha", x) for d in DATASETS]) for x in xs]
        ax.plot(xs, ys, marker="s", ms=3.1, color=METHOD_COLORS[method])
    ax.set(xlabel="Dirichlet concentration $\\alpha$", ylabel="Macro-F1 score $\\uparrow$", xticks=xs)
    ax.set_title("Statistical heterogeneity", pad=3); style_axis(ax); panel_label(ax, "(b)")

    # (c) participation composite bars + line
    ax = axes[0, 2]; xs = [0.3, 0.6, 0.8, 1.0]
    base = [np.mean([stress_value(runs, d, "fedgcc", "participation", x) for d in DATASETS]) for x in xs]
    ours = [np.mean([stress_value(runs, d, "fedgeski", "participation", x) for d in DATASETS]) for x in xs]
    width = 0.11
    ax.bar(np.asarray(xs) - width/2, base, width=width, color=METHOD_COLORS["fedgcc"], alpha=0.72, edgecolor="#333333", lw=0.5)
    ax.bar(np.asarray(xs) + width/2, ours, width=width, color=METHOD_COLORS["fedgeski"], alpha=0.78, edgecolor="#333333", lw=0.5)
    gain = (np.asarray(ours) - np.asarray(base)) * 100
    ax2 = ax.twinx(); ax2.plot(xs, gain, color="#4D4D4D", marker="D", ms=3.2, lw=1.1); ax2.set_ylabel("F1 gain (percentage points) $\\uparrow$"); ax2.tick_params(direction="in", width=0.7, length=3)
    ax.set(xlabel="Client participation ratio", ylabel="Macro-F1 score $\\uparrow$", xticks=xs)
    ax.set_title("Sparse participation benefit", pad=3); style_axis(ax); panel_label(ax, "(c)")

    # (d) error quantile curve from final predictions
    ax = axes[1, 0]
    quantiles = np.linspace(0, 1, 101)
    for method in METHODS:
        errors = []
        for dataset in DATASETS:
            for seed in range(5):
                pred = MODELS_DIR / f"{dataset}__{method}__s{seed}__a0.3__lag2__p0.6.predictions.npz"
                if pred.exists():
                    data = np.load(pred); errors.extend((1 - data["probabilities"][np.arange(len(data["y_true"])), data["y_true"]]).tolist())
        if errors:
            ax.plot(quantiles, np.quantile(errors, quantiles), color=METHOD_COLORS[method])
    ax.set(xlabel="Sample quantile", ylabel="True-class probability error $\\downarrow$", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("Error quantile profile", pad=3); style_axis(ax); panel_label(ax, "(d)")

    # (e) relative degradation from nominal setting, horizontal range bars.
    ax = axes[1, 1]
    degradations = {m: [] for m in METHODS}
    for method in METHODS:
        nominal = np.mean([stress_value(runs, d, method, "max_staleness", 2) for d in DATASETS])
        settings = [
            np.mean([stress_value(runs, d, method, "max_staleness", 4) for d in DATASETS]),
            np.mean([stress_value(runs, d, method, "dirichlet_alpha", 0.1) for d in DATASETS]),
            np.mean([stress_value(runs, d, method, "participation", 0.3) for d in DATASETS]),
        ]
        degradations[method] = [100 * (nominal - v) / max(nominal, 1e-8) for v in settings]
    for yi, method in enumerate(METHODS):
        vals = degradations[method]
        ax.hlines(yi, min(vals), max(vals), color=METHOD_COLORS[method], lw=3, alpha=0.55)
        ax.scatter(vals, [yi] * 3, color=METHOD_COLORS[method], s=20, edgecolor="#222222", lw=0.45, zorder=3)
    ax.axvline(0, color="#333333", lw=0.7, ls="--")
    ax.set_yticks(range(len(METHODS)), [METHOD_LABELS[m] for m in METHODS]); angle_method_ticks(ax, "y"); ax.invert_yaxis()
    ax.set(xlabel="Relative degradation (%) $\\downarrow$")
    ax.set_title("Worst-case degradation span", pad=3); style_axis(ax); panel_label(ax, "(e)")

    # (f) win-rate matrix across 26 unique stress scenarios.
    ax = axes[1, 2]
    scenarios = []
    for dataset in DATASETS:
        for key, values in (("max_staleness", [0,1,2,3,4]), ("dirichlet_alpha", [0.1,0.3,0.6,1.0]), ("participation", [0.3,0.6,0.8,1.0])):
            for value in values:
                scenario = [stress_value(runs, dataset, m, key, value) for m in METHODS]
                if np.all(np.isfinite(scenario)): scenarios.append(scenario)
    scenarios = np.asarray(scenarios)
    pairwise = np.zeros((len(METHODS), len(METHODS)))
    for i in range(len(METHODS)):
        for j in range(len(METHODS)): pairwise[i,j] = np.mean(scenarios[:, i] > scenarios[:, j]) if i != j else 0.5
    im = ax.imshow(pairwise, cmap=POSITIVE_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(METHODS)), [METHOD_LABELS[m] for m in METHODS]); angle_method_ticks(ax, "x")
    ax.set_yticks(range(len(METHODS)), [METHOD_LABELS[m] for m in METHODS]); angle_method_ticks(ax, "y")
    for i in range(len(METHODS)):
        for j in range(len(METHODS)): ax.text(j, i, f"{pairwise[i,j]:.2f}", ha="center", va="center", fontsize=4.8)
    ax.set_title("Pairwise stress-test win rate $\\uparrow$", pad=3); ax.tick_params(direction="out", length=2); panel_label(ax, "(f)")

    compact = {"fedgcc":"Fed-GCC", "afcl_csc":"AFCL-CSC", "fedavgm":"FAvgM",
               "fedadam":"FAdam", "fedyogi":"FYogi", "fedasync":"FAsync",
               "fedbuff":"FBuff", "glfc":"GLFC", "evofedids":"EvoIDS",
               "fedta":"FTA", "fedagc":"FAGC", "fedgeski":"FedGeSKI"}
    line_handles = [Line2D([0], [0], color=METHOD_COLORS[m], marker="o", markersize=3.4,
                           lw=1.8, label=compact[m]) for m in METHODS]
    circle_handles = [Line2D([0], [0], marker="o", linestyle="None", markersize=5.2,
                             markerfacecolor=METHOD_COLORS[m], markeredgecolor="#222222",
                             markeredgewidth=0.45, label=compact[m]) for m in METHODS]
    fig.legend(handles=line_handles, loc="lower center", ncol=12, frameon=False,
               bbox_to_anchor=(0.5, 0.055), fontsize=7.5, columnspacing=0.30,
               handletextpad=0.20, handlelength=1.05)
    fig.legend(handles=circle_handles, loc="lower center", ncol=12, frameon=False,
               bbox_to_anchor=(0.5, 0.012), fontsize=7.5, columnspacing=0.30,
               handletextpad=0.22, handlelength=0.75)
    save_figure(fig, "robustness_dashboard.pdf")


def calibration_curve(y: np.ndarray, probs: np.ndarray, bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    conf = probs.max(axis=1); pred = probs.argmax(axis=1); correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); centers, accs, counts = [], [], []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (conf >= left) & (conf < right if right < 1 else conf <= right)
        if mask.any(): centers.append(conf[mask].mean()); accs.append(correct[mask].mean()); counts.append(mask.sum())
    return np.asarray(centers), np.asarray(accs), np.asarray(counts)


def plot_diagnostics(runs: List[dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 5.10))
    fig.subplots_adjust(left=0.078, right=0.985, bottom=0.17, top=0.925, wspace=0.48, hspace=0.45)

    # (a) reliability diagram on pooled tests.
    ax = axes[0, 0]; ax.plot([0,1], [0,1], ls="--", color="#555555", lw=0.8)
    for method in METHODS:
        ys, ps = [], []
        for dataset in DATASETS:
            pred = MODELS_DIR / f"{dataset}__{method}__s0__a0.3__lag2__p0.6.predictions.npz"
            if pred.exists():
                d = np.load(pred); ys.append(d["y_true"]); ps.append(d["probabilities"])
        if ys:
            c, a, _ = calibration_curve(np.concatenate(ys), np.concatenate(ps))
            ax.plot(c, a, marker="o", ms=2.8, color=METHOD_COLORS[method])
    ax.set(xlabel="Mean confidence", ylabel="Empirical accuracy", xlim=(0,1), ylim=(0,1))
    ax.set_title("Reliability diagram", pad=3); style_axis(ax); panel_label(ax, "(a)")

    # (b) pooled confusion matrix of FedGeSKI.
    ax = axes[0, 1]; cm = np.zeros((6,6))
    for dataset in DATASETS:
        pred = MODELS_DIR / f"{dataset}__fedgeski__s0__a0.3__lag2__p0.6.predictions.npz"
        d = np.load(pred); cm += confusion_matrix(d["y_true"], d["probabilities"].argmax(1), labels=np.arange(6), normalize=None)
    cm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm, cmap=POSITIVE_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(6), [f"C{i}" for i in range(6)]); ax.set_yticks(range(6), [f"C{i}" for i in range(6)])
    ax.set(xlabel="Predicted class", ylabel="True class")
    for i in range(6):
        for j in range(6): ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=5.8)
    ax.set_title("Normalized confusion matrix", pad=3); ax.tick_params(direction="out", length=2); panel_label(ax, "(b)")

    # (c) latent PCA and ideal anchors.
    ax = axes[0, 2]
    pred = np.load(MODELS_DIR / "edgeiiot__fedgeski__s0__a0.3__lag2__p0.6.predictions.npz")
    rng = np.random.default_rng(2026); idx = np.concatenate([rng.choice(np.where(pred["y_true"] == c)[0], 100, replace=False) for c in range(6)])
    ckpt = __import__("torch").load(MODELS_DIR / "edgeiiot__fedgeski__s0__a0.3__lag2__p0.6.pt", map_location="cpu")
    combined = np.vstack([pred["embeddings"][idx], ckpt["anchors"].numpy()])
    xy = PCA(n_components=2, random_state=2026).fit_transform(combined)
    class_colors = ["#FF6666", "#FFAA53", "#50CC55", "#3399FF", "#6666FF", "#9933FF"]
    for c in range(6):
        mask = pred["y_true"][idx] == c
        ax.scatter(xy[:len(idx)][mask,0], xy[:len(idx)][mask,1], s=6, alpha=0.32, color=class_colors[c], linewidth=0)
    anchors = xy[len(idx):]
    ax.scatter(anchors[:,0], anchors[:,1], marker="*", s=75, c=class_colors, edgecolor="#111111", lw=0.55, zorder=4)
    ax.set(xlabel="Principal component 1 (a.u.)", ylabel="Principal component 2 (a.u.)")
    ax.set_title("Simplex-organized latent space", pad=3); style_axis(ax); panel_label(ax, "(c)")

    # (d) signed backward transfer heat map (swapped with coverage diagnostic).
    ax = axes[1, 0]
    matrix = []
    for method in METHODS:
        vals = [np.asarray(r["bwt_per_class"]) for r in runs if is_main(r) and r["config"]["method"] == method]
        matrix.append(np.mean(vals, axis=0))
    matrix = np.asarray(matrix); vmax = max(abs(matrix.min()), abs(matrix.max()))
    im = ax.imshow(matrix, cmap=SIGNED_CMAP, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), aspect="auto")
    ax.set_xticks(range(6), [f"C{i}" for i in range(6)]); ax.set_yticks(range(len(METHODS)), [METHOD_LABELS[m] for m in METHODS]); angle_method_ticks(ax, "y")
    for i in range(len(METHODS)):
        for j in range(6): ax.text(j, i, f"{matrix[i,j]:+.2f}", ha="center", va="center", fontsize=5.5)
    ax.set_title("Class-wise backward transfer $\\uparrow$", pad=3); ax.tick_params(direction="out", length=2); panel_label(ax, "(d)")

    # (e) historical-gradient coverage of the recent-update subspace.
    ax = axes[1, 1]
    lags, coverage, residual, forgetting = [], [], [], []
    paired_coverage, paired_forgetting = [], []
    for lag in (0, 1, 2, 3, 4):
        chosen = [r for r in runs if r["config"]["method"] == "fedgeski"
                  and r["config"].get("variant", "full") == "full"
                  and r["config"]["seed"] in range(5) and r["config"]["max_staleness"] == lag
                  and r["config"]["dirichlet_alpha"] == 0.3 and r["config"]["participation"] == 0.6]
        vals = [rec.get("history_subspace_coverage", np.nan) for r in chosen for rec in r["records"]]
        res = [rec.get("history_subspace_residual", np.nan) for r in chosen for rec in r["records"]]
        vals = np.asarray(vals, dtype=float); vals = vals[np.isfinite(vals)]
        res = np.asarray(res, dtype=float); res = res[np.isfinite(res)]
        if chosen and vals.size and res.size:
            lags.append(lag); coverage.append(float(np.mean(vals))); residual.append(float(np.mean(res)))
            forgetting.append(float(np.mean([r["final"]["average_forgetting"] for r in chosen])))
            for run in chosen:
                run_cov = np.asarray(
                    [rec.get("history_subspace_coverage", np.nan) for rec in run["records"]], dtype=float
                )
                run_cov = run_cov[np.isfinite(run_cov)]
                if run_cov.size:
                    paired_coverage.append(float(run_cov.mean()))
                    paired_forgetting.append(float(run["final"]["average_forgetting"]))
    ax.plot(lags, coverage, marker="o", color="#FF6666", label="Gradient coverage")
    ax.plot(lags, residual, marker="^", ls="--", color="#FFAA53", label="Relative residual")
    ax.set(xlabel="Maximum model age (rounds)", ylabel="Coverage $\\uparrow$ / residual $\\downarrow$", xticks=lags)
    ax.tick_params(axis="y", labelrotation=32)
    ax2 = ax.twinx(); ax2.plot(lags, forgetting, marker="s", color="#3399FF", label="Forgetting")
    ax2.set_ylabel(""); ax2.tick_params(direction="in", width=0.7, length=3, axis="y", labelrotation=32)
    ax2.text(0.98, 0.94, "Forgetting $\\downarrow$", transform=ax2.transAxes,
             ha="right", va="top", color="#3399FF", fontsize=6.3,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.7))
    ax.legend(ax.get_lines(), [line.get_label() for line in ax.get_lines()], frameon=True,
              loc="center", bbox_to_anchor=(0.57, 0.57), fontsize=6.1,
              handlelength=1.4, handletextpad=0.3, facecolor="white", edgecolor="none")
    if len(paired_coverage) >= 3:
        rho = pd.Series(paired_coverage).rank().corr(pd.Series(paired_forgetting).rank())
        ax.text(0.03, 0.04, f"Spearman $\\rho$={rho:+.2f}", transform=ax.transAxes, fontsize=6.6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.6))
    ax.set_title("Subspace diagnostic", pad=3); style_axis(ax); panel_label(ax, "(e)")

    # (f) cross-version anchor alignment grouped by model age.
    ax = axes[1, 2]
    by_age = {age: [] for age in range(5)}
    for run in runs:
        if run["config"]["method"] == "fedgeski" and run["config"].get("variant", "full") == "full":
            for rec in run["records"]:
                for age, value in rec.get("anchor_alignment_by_staleness", {}).items():
                    if int(age) in by_age: by_age[int(age)].append(float(value))
    ages = [age for age, values in by_age.items() if values]
    values = [by_age[age] for age in ages]
    violin = ax.violinplot(values, positions=ages, widths=0.72, showmeans=True, showextrema=True)
    for body in violin["bodies"]:
        body.set_facecolor("#FF6666"); body.set_edgecolor("#333333"); body.set_alpha(0.62)
    ax.axhline(0, color="#777777", lw=0.7, ls="--")
    ax.set(xlabel="Client-model age (rounds)", ylabel="Assigned-anchor cosine $\\uparrow$", xticks=ages)
    ax.set_title("Cross-version latent alignment", pad=3); style_axis(ax); panel_label(ax, "(f)")

    handles = [Line2D([0], [0], color=METHOD_COLORS[m], marker="o", ms=3, lw=1.7, label=METHOD_LABELS[m]) for m in METHODS]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.030), fontsize=7.5, columnspacing=0.55,
               handletextpad=0.25, handlelength=1.20)
    save_figure(fig, "mechanism_diagnostics.pdf")


def main() -> None:
    runs = load_json_runs()
    plot_main_dynamics(runs)
    plot_robustness(runs)
    plot_diagnostics(runs)


if __name__ == "__main__":
    main()
