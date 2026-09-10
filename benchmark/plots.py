from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

REDUCERS = ["PCA", "KernelPCA", "VAE", "Isomap", "MDS"]
LEVELS = ["k_minus_1", "25_percent", "50_percent"]
CONDITIONS = [
    "None__original",
    *[f"{r}__{level}" for r in REDUCERS for level in LEVELS],
]
COLORS = {
    "None": "#1f77b4",
    "PCA": "#ff7f0e",
    "KernelPCA": "#2ca02c",
    "VAE": "#d62728",
    "Isomap": "#9467bd",
    "MDS": "#8c564b",
}
DISPLAY_REDUCER = {
    "PCA": "PCA",
    "KernelPCA": "Kernel PCA",
    "VAE": "VAE",
    "Isomap": "Isomap",
    "MDS": "MDS",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.8,
            "font.size": 10,
            "axes.labelsize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(fig: plt.Figure, output: Path, number: int | str) -> None:
    stem = output / f"figure_{number}"
    fig.savefig(
        stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _condition_label(condition: str) -> str:
    if condition == "None__original":
        return "No Reduction"
    reducer, level = condition.split("__", 1)
    plot_level = {"k_minus_1": "k-1", "25_percent": "25%", "50_percent": "50%"}[level]
    return f"{DISPLAY_REDUCER[reducer]} {plot_level}"


def configuration_units(units: pd.DataFrame) -> pd.DataFrame:
    synthetic = units[units["scope"] == "synthetic"]
    keys = [
        "family",
        "n_clusters",
        "n_features",
        "n_samples",
        "clusterer",
        "reducer",
        "reduction_level",
        "condition",
    ]
    return synthetic.groupby(keys, as_index=False)["ari"].mean()


def _boxplot_axis(data: list[np.ndarray]) -> tuple[np.ndarray, tuple[float, float]]:
    smallest = float(min((float(values.min()) for values in data if len(values))))
    lowest_tick = min(-0.2, float(np.floor(round(smallest, 10) * 10.0) / 10.0))
    return (np.arange(lowest_tick, 1.0001, 0.1), (lowest_tick - 0.06, 1.06))


def _paper_boxplot(
    units: pd.DataFrame, scope: str, clusterer: str, output: Path, number: int | str
) -> None:
    source = (
        configuration_units(units)
        if scope == "synthetic"
        else units[units["scope"] == scope]
    )
    block = source[source["clusterer"] == clusterer]
    data = [
        block.loc[block["condition"] == condition, "ari"].to_numpy()
        for condition in CONDITIONS
    ]
    if any((len(values) == 0 for values in data)):
        raise RuntimeError(f"missing boxplot values for {scope}/{clusterer}")
    fig, ax = plt.subplots(figsize=(9.2, 4.15))
    artists = ax.boxplot(
        data,
        widths=0.58,
        patch_artist=True,
        whis=1.5,
        medianprops={"color": "black", "linewidth": 1.1},
        whiskerprops={"color": "#555555", "linewidth": 0.9},
        capprops={"color": "#555555", "linewidth": 0.9},
    )
    for index, (box, flier) in enumerate(
        zip(artists["boxes"], artists["fliers"], strict=True)
    ):
        reducer = "None" if index == 0 else CONDITIONS[index].split("__", 1)[0]
        color = COLORS[reducer]
        box.set(facecolor=color, edgecolor="#444444", linewidth=0.7, alpha=0.9)
        flier.set(
            marker="o",
            markersize=3.2,
            markerfacecolor=color,
            markeredgecolor=color,
            alpha=0.9,
        )
    ax.set_xticks(
        np.arange(1, len(CONDITIONS) + 1), [_condition_label(c) for c in CONDITIONS]
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("ARI score")
    ax.set_xlabel("Dimensionality Reduction Methods")
    ticks, limits = _boxplot_axis(data)
    ax.set_yticks(ticks)
    ax.set_ylim(*limits)
    ax.grid(False)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.97, bottom=0.3)
    _save_figure(fig, output, number)


def _rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    color: str,
) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008",
            facecolor=color,
            edgecolor="#0a2a3a",
            linewidth=1.2,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "color": "#0b5d87", "lw": 1.1},
    )


def _figure_two(units: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    blue = "#0b5d87"
    for i in range(4):
        for j in range(4):
            ax.add_patch(
                Rectangle(
                    (0.07 + j * 0.022, 0.72 + (3 - i) * 0.022),
                    0.021,
                    0.021,
                    facecolor=str(0.55 + 0.12 * ((i + j) % 3)),
                    edgecolor="white",
                    lw=0.5,
                )
            )
    ax.text(0.113, 0.825, "Features", ha="center", fontsize=9)
    ax.text(0.052, 0.765, "Objects", ha="center", va="center", rotation=90, fontsize=9)
    _rounded_box(ax, (0.2, 0.735), (0.14, 0.065), "Data\nnormalization", "#d8d8d8")
    ax.add_patch(
        FancyBboxPatch(
            (0.42, 0.575),
            0.28,
            0.335,
            boxstyle="round,pad=0.02",
            facecolor="none",
            edgecolor="#0a2a3a",
            linestyle=(0, (4, 3)),
            linewidth=1.2,
        )
    )
    ax.text(0.56, 0.885, "Dimensionality reduction", ha="center", fontsize=10)
    ax.add_patch(
        Rectangle((0.445, 0.785), 0.23, 0.08, facecolor="#e6e6e6", edgecolor="#0a2a3a")
    )
    ax.text(0.56, 0.846, "Linear methods", ha="center", fontsize=8)
    _rounded_box(ax, (0.525, 0.792), (0.07, 0.03), "PCA", "#b9ddec")
    ax.add_patch(
        Rectangle((0.445, 0.68), 0.23, 0.08, facecolor="#e6e6e6", edgecolor="#0a2a3a")
    )
    ax.text(0.56, 0.741, "Non-linear methods", ha="center", fontsize=8)
    for x, label in ((0.452, "KPCA"), (0.525, "ISOMAP"), (0.607, "MDS")):
        _rounded_box(ax, (x, 0.687), (0.062, 0.033), label, "#f6dfd2")
    ax.add_patch(
        Rectangle((0.445, 0.585), 0.23, 0.08, facecolor="#e6e6e6", edgecolor="#0a2a3a")
    )
    ax.text(0.56, 0.646, "Auto-encoders", ha="center", fontsize=8)
    _rounded_box(ax, (0.525, 0.592), (0.07, 0.03), "VAE", "#d5efc4")
    ax.add_patch(
        FancyBboxPatch(
            (0.73, 0.1),
            0.2,
            0.3,
            boxstyle="round,pad=0.02",
            facecolor="none",
            edgecolor="black",
            linestyle=(0, (4, 3)),
            linewidth=1.2,
        )
    )
    ax.text(0.83, 0.375, "Clustering algorithms", ha="center", fontsize=10)
    for y, label, color in (
        (0.305, "K-MEANS", "#e3b4df"),
        (0.245, "AGGLOMERATIVE\nCLUSTERING", "#f2a274"),
        (0.185, "GAUSSIAN\nMIXTURE", "#55bfe6"),
        (0.125, "OPTICS", "#81d568"),
    ):
        _rounded_box(ax, (0.755, y), (0.15, 0.048), label, color)
    ax.add_patch(
        FancyBboxPatch(
            (0.44, 0.1),
            0.18,
            0.3,
            boxstyle="round,pad=0.02",
            facecolor="none",
            edgecolor="#0a2a3a",
            linestyle=(0, (4, 3)),
            linewidth=1.2,
        )
    )
    ax.text(0.53, 0.375, "Computing ARI", ha="center", fontsize=10)
    _rounded_box(ax, (0.475, 0.285), (0.12, 0.04), "Cluster labels", "#bcbcbc")
    _rounded_box(ax, (0.475, 0.175), (0.12, 0.04), "Ground truth", "#bcbcbc")
    ax.plot([0.535, 0.535], [0.215, 0.285], color=blue, lw=1.0)
    ax.plot([0.535, 0.515], [0.25, 0.25], color=blue, lw=1.0)
    ax.text(0.508, 0.25, "ARI", ha="right", va="center", fontsize=8)
    order = ["MDS", "Isomap", "VAE", "KernelPCA", "PCA", "None"]
    synthetic = units[units["scope"] == "synthetic"]
    cells = synthetic.groupby(
        ["family", "reducer", "reduction_level", "clusterer"], as_index=False
    )["ari"].mean()
    means = (
        cells.groupby(["family", "reducer"])["ari"]
        .max()
        .groupby("reducer")
        .mean()
        .reindex(order)
    )
    bar_ax = fig.add_axes([0.095, 0.125, 0.265, 0.245])
    bar_ax.barh(
        ["MDS", "Isomap", "VAE", "KernelPCA", "PCA", "No reduction"],
        means.to_numpy(),
        color=[COLORS[name] for name in order],
    )
    bar_ax.set_xlabel("ARI", fontsize=8)
    bar_ax.set_title("ARI Comparison", fontsize=9)
    bar_ax.tick_params(labelsize=7)
    bar_ax.set_xlim(0, float(means.max()) * 1.25)
    for y, value in enumerate(means):
        bar_ax.text(value + 0.005, y, f"{value:.2f}", va="center", fontsize=6)
    _arrow(ax, (0.165, 0.767), (0.2, 0.767))
    ax.plot([0.34, 0.375], [0.767, 0.767], color=blue, lw=1.1)
    ax.plot([0.375, 0.375], [0.625, 0.825], color=blue, lw=1.1)
    for y_branch in (0.825, 0.72, 0.625):
        _arrow(ax, (0.375, y_branch), (0.445, y_branch))
    for y in (0.825, 0.72, 0.625):
        ax.plot([0.675, 0.76], [y, y], color=blue, lw=1.1)
    ax.plot([0.76, 0.76], [0.625, 0.825], color=blue, lw=1.1)
    ax.plot([0.76, 0.86, 0.86], [0.625, 0.625, 0.44], color=blue, lw=1.1)
    _arrow(ax, (0.86, 0.44), (0.86, 0.405))
    ax.text(0.875, 0.53, "Reduced\ndata", ha="left", va="center", fontsize=8)
    ax.plot([0.27, 0.27, 0.8, 0.8], [0.735, 0.49, 0.49, 0.44], color=blue, lw=1.1)
    _arrow(ax, (0.8, 0.44), (0.8, 0.405))
    ax.text(0.53, 0.505, "Without dimensionality reduction", ha="center", fontsize=8)
    _arrow(ax, (0.705, 0.25), (0.648, 0.25))
    ax.text(
        0.676, 0.268, "Obtained\nclusterings", ha="center", va="bottom", fontsize=7.5
    )
    _arrow(ax, (0.418, 0.25), (0.375, 0.25))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _save_figure(fig, output, 2)
