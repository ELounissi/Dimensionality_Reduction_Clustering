"""Generate the paper's numerical tables and figures from per-run ARI records."""

from pathlib import Path
import shutil
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from . import analysis as a
from . import plots as p

ROOT = Path(__file__).resolve().parents[1]


def pcell(row, signed=False, holmkey=None):
    value = f"{max(0.001, row.p_raw):.3f}"
    if signed:
        value = (
            "+" if row.mean_delta_ari > 0 else "$-$" if row.mean_delta_ari < 0 else ""
        ) + value
    if row.p_raw < 0.05:
        value = r"{\bfseries\boldmath " + value + "}"
    return value


DISPLAY = {"None": "No reduction", "KernelPCA": "Kernel PCA", "KMeans": "$k$-means"}
LEVEL = {"k_minus_1": "$k-1$", "25_percent": "25\\%", "50_percent": "50\\%"}
TLABELS = [
    "tab:uci_datasets",
    "tab:kmeans_aggregate",
    "tab:agglomerative_aggregate",
    "tab:gmm_aggregate",
    "tab:optics_aggregate",
    "tab:ari_circles",
    "tab:ari_moons",
    "tab:ari_rodriguez",
    "tab:ari_repliclust",
    "tab:kmeans_real",
    "tab:agglomerative_real",
    "tab:gaussian_real",
    "tab:optics_real",
    "tab:Wilcoxon",
]


def display(s):
    return DISPLAY.get(s, s)


def tabular(headers, rows, spec=None):
    return (
        "\\begin{tabular}{"
        + (spec or "l" + "r" * (len(headers) - 1))
        + "}\n\\toprule\n"
        + " & ".join(headers)
        + " \\\\"
        + "\n\\midrule\n"
        + "\n".join((" & ".join(row) + " \\\\" for row in rows))
        + "\n\\bottomrule\n\\end{tabular}"
    )


def wide(rows, baseline=True, reducers=None):
    reducers = a.RS if reducers is None else reducers
    offset = 3 if baseline else 2
    title = (lambda x: r"\textbf{" + x + "}") if baseline else str
    head = " & " + ((title("No Reduction") + " & ") if baseline else "")
    head += (
        " & ".join(r"\multicolumn{3}{c}{" + title(display(r)) + "}" for r in reducers)
        + r" \\"
        + "\n"
    )
    head += (
        " ".join(
            r"\cmidrule(lr){"
            + str(offset + 3 * i)
            + "-"
            + str(offset + 3 * i + 2)
            + "}"
            for i in range(len(reducers))
        )
        + "\n"
    )
    head += (
        title("Algorithms" if baseline else "Clusterer")
        + (" & " if baseline else "")
        + " & "
    )
    head += (
        " & ".join(title(LEVEL[level]) for r in reducers for level in a.LS)
        + r" \\"
        + "\n"
        + r"\midrule"
        + "\n"
    )
    return (
        r"\begin{tabular}{l"
        + "c" * (3 * len(reducers) + int(baseline))
        + "}\n"
        + r"\toprule"
        + "\n"
        + head
        + "\n".join(" & ".join(row) + r" \\" for row in rows)
        + "\n"
        + r"\bottomrule\end{tabular}"
    )


def aggregate_table(rows):
    head = r"\begin{tabular}{l l cc cc}" + "\n" + r"\toprule" + "\n"
    head += (
        r"\textbf{Method} & \textbf{Reduction} & \multicolumn{2}{c}{\textbf{Percentage of Wins}} & \multicolumn{2}{c}{\textbf{Average win/loss (-) percentage}} \\"
        + "\n"
    )
    head += r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}" + "\n"
    head += (
        r"& & \textbf{Synthetic Data} & \textbf{Real-world Data} & \textbf{Synthetic Data} & \textbf{Real-world Data} \\"
        + "\n"
        + r"\midrule"
        + "\n"
    )
    body = []
    for i, row in enumerate(rows):
        row = row.copy()
        row[0] = r"\multirow{3}{*}{" + row[0] + "}" if i % 3 == 0 else ""
        if i and i % 3 == 0:
            body.append(r"\cmidrule(lr){1-6}")
        body.append(" & ".join(row) + r" \\")
    return head + "\n".join(body) + "\n" + r"\bottomrule\end{tabular}"


def mean_sd(mean, sd, bold=False):
    value = r"\shortstack{" + f"{mean:.3f}" + r"\\($\pm$ " + f"{sd:.3f}" + ")}"
    return r"{\bfseries " + value + "}" if bold else value


def artifacts(u, output):
    def save(frame, name):
        a.save(frame, output / "tables" / f"{name}.csv")

    tables = {}
    caps = {}
    real = u[u.scope == "real"]
    manifest = (
        real[["dataset_id", "display_name", "n_samples", "n_features", "n_clusters"]]
        .drop_duplicates()
        .sort_values("dataset_id")
    )
    save(manifest, "table_1")
    tables["tab:uci_datasets"] = tabular(
        ["Dataset", "Objects", "Features", "Clusters"],
        [
            [r.display_name, str(r.n_samples), str(r.n_features), str(r.n_clusters)]
            for r in manifest.itertuples()
        ],
    )
    base = u[u.condition == a.CONDS[0]][["dataset_id", "clusterer", "ari"]].rename(
        columns={"ari": "baseline"}
    )
    delta = u.merge(base, on=["dataset_id", "clusterer"], validate="many_to_one")
    delta["delta_ari"] = delta.ari - delta.baseline
    save(delta, "paired_dataset_scores")
    for i, c in enumerate(a.CS, 2):
        sub = delta[(delta.clusterer == c) & (delta.condition != a.CONDS[0])]
        summ = (
            sub.groupby(["condition", "scope"])
            .agg(
                mean_delta=("delta_ari", "mean"),
                wins=("delta_ari", lambda x: 100 * (x > 0).mean()),
            )
            .reset_index()
        )
        save(summ, f"table_{i}")
        rows = []
        for cond in a.CONDS[1:]:
            q = summ[summ.condition == cond].set_index("scope")
            r, l = cond.split("__")
            rows.append(
                [
                    display(r),
                    LEVEL[l],
                    f"{q.loc['synthetic', 'wins']:.1f}",
                    f"{q.loc['real', 'wins']:.1f}",
                    f"{100 * q.loc['synthetic', 'mean_delta']:+.2f}",
                    f"{100 * q.loc['real', 'mean_delta']:+.2f}",
                ]
            )
        label = TLABELS[i - 1]
        tables[label] = aggregate_table(rows)
    for i, f in enumerate(["circles", "moons", "rsg", "repliclust"], 1):
        stats = (
            u[u.family == f]
            .groupby(["clusterer", "condition"])
            .ari.agg(["mean", "std", "count"])
            .reset_index()
        )
        save(stats, f"A{i}")
        rows = []
        for c in a.CS:
            z = stats[stats.clusterer == c].set_index("condition").reindex(a.CONDS)
            maximum = z["mean"].max()
            rows.append(
                [display(c)]
                + [
                    mean_sd(
                        x["mean"],
                        x["std"],
                        np.isclose(x["mean"], maximum, atol=1e-12, rtol=0),
                    )
                    for _, x in z.iterrows()
                ]
            )
        label = TLABELS[i + 4]
        tables[label] = wide(rows)
        {
            "circles": "Circles",
            "moons": "Moons",
            "rsg": "Rodriguez Structured Gaussian",
            "repliclust": "Repliclust",
        }[f]
    for i, c in enumerate(a.CS, 5):
        z = (
            real[real.clusterer == c]
            .pivot(index="display_name", columns="condition", values="ari")
            .reindex(index=manifest.display_name, columns=a.CONDS)
        )
        save(z.reset_index(), f"A{i}")
        rows = []
        for name, x in z.iterrows():
            rows.append(
                [name]
                + [
                    "\\textbf{" + f"{v:.3f}" + "}"
                    if np.isclose(v, x.max(), atol=1e-12, rtol=0)
                    else f"{v:.3f}"
                    for v in x
                ]
            )
        label = TLABELS[i + 4]
        tables[label] = wide(rows).replace("Algorithms", "Dataset", 1)
    stats = {
        9: a.versus_baseline(u, "synthetic"),
        10: a.versus_baseline(u, "real"),
        11: a.pairwise(u, "reducer"),
        12: a.pairwise(u, "clusterer"),
    }
    for i, s in stats.items():
        save(s, f"A{i}")
    for i in [9, 10]:
        s = stats[i]
        rows = []
        for c in a.CS:
            z = s[s.clusterer == c].set_index("condition").reindex(a.CONDS[1:])
            rows.append([display(c)] + [pcell(r, signed=True) for r in z.itertuples()])
        label = "tab:Wilcoxon" if i == 10 else "tab:Wilcoxon_synthetic"
        tables[label] = wide(rows, False)
    for i, order, label in [
        (11, ["None"] + a.RS, "tab:reducer_comparison"),
        (12, a.CS, "tab:clusterer_comparison"),
    ]:
        rows = []
        for j, r in enumerate(order):
            row = [display(r)]
            for k, c in enumerate(order):
                if k >= j:
                    row.append("--")
                    continue
                x = stats[i][(stats[i].row == c) & (stats[i].column == r)].iloc[0]
                x = x.copy()
                x["mean_delta_ari"] *= -1
                row.append(pcell(x, True))
            rows.append(row)
        tables[label] = tabular([""] + [display(r) for r in order], rows)
    p._setup_style()
    shutil.copy2(ROOT / "assets/figure_1.pdf", output / "figures/figure_1.pdf")
    p._figure_two(u, output / "figures")
    for scope, start in [("synthetic", 3), ("real", 7)]:
        for i, c in enumerate(a.CS, start):
            p._paper_boxplot(u, scope, c, output / "figures", i)
    winners = (
        stats[10]
        .sort_values("mean_delta_ari", ascending=False, kind="stable")
        .groupby(["reducer", "clusterer"], sort=False)
        .head(1)
    )
    save(winners, "figure_A1_levels")
    matrix = winners.pivot(
        index="reducer", columns="clusterer", values="mean_delta_ari"
    ).reindex(index=a.RS, columns=a.CS)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    lim = max(abs(matrix.to_numpy()).max(), 0.16)
    im = ax.imshow(
        matrix,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
        aspect="auto",
    )
    ax.set_xticks(range(4), ["$k$-means", "AHC", "GMM", "OPTICS"])
    ax.set_yticks(range(5), [display(r) for r in a.RS])
    ax.tick_params(length=0)
    for i, r in enumerate(a.RS):
        for j, c in enumerate(a.CS):
            x = winners[(winners.reducer == r) & (winners.clusterer == c)].iloc[0]
            v = x.mean_delta_ari
            ax.text(
                j,
                i,
                f"{v:+.3f}" + (" *" if x.p_raw < 0.05 else ""),
                ha="center",
                va="center",
                color="white" if abs(v) > 0.08 else "#111827",
                fontsize=13,
            )
    ax.set_title(
        "Real data: maximum mean ARI gain across three retention levels",
        pad=16,
        fontsize=13,
    )
    bar = fig.colorbar(im, ax=ax, pad=0.04)
    bar.set_label("Mean paired ARI change")
    fig.text(
        0.12,
        0.015,
        "* Nominal p < 0.05 at the displayed level; none survives Holm correction.",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.15, right=0.9, top=0.87, bottom=0.12)
    p._save_figure(fig, output / "figures", 11)
    for label, body in tables.items():
        name = label.replace(":", "_")
        (output / "tables" / f"{name}.tex").write_text(body + "\n", encoding="utf8")
    return (tables, caps, stats)


def markdown(headers, rows):
    return (
        "| "
        + " | ".join(headers)
        + " |\n| "
        + " | ".join(["---"] * len(headers))
        + " |\n"
        + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)
        + "\n"
    )


def plain(value):
    return display(value).replace("$", "")


def signed_p(row):
    value = (
        "+" if row.mean_delta_ari > 0 else "-" if row.mean_delta_ari < 0 else ""
    ) + f"{max(0.001, row.p_raw):.3f}"
    return "**" + value + "**" if row.p_raw < 0.05 else value


def write_results(u, output, stats):
    conditions = ["No reduction"] + [
        plain(r)
        + " "
        + {"k_minus_1": "k-1", "25_percent": "25%", "50_percent": "50%"}[l]
        for r in a.RS
        for l in a.LS
    ]
    text = [
        "# Paper results\n",
        "[Project home](https://github.com/ELounissi/Dimensionality_Reduction_Clustering) | [Guided notebook](../benchmark.ipynb)\n",
        "All CSV files retain full precision. The tables below use the manuscript's display precision. Bold indicates a row maximum in ARI tables or nominal p < 0.05 in statistical tables.\n",
    ]
    for scope, number in (("real", 10), ("synthetic", 9)):
        rows = []
        for c in a.CS:
            q = (
                stats[number][stats[number].clusterer == c]
                .sort_values("mean_ari", ascending=False, kind="stable")
                .iloc[0]
            )
            rows.append(
                [
                    plain(c),
                    f"{q.baseline_ari:.4f}",
                    plain(q.reducer),
                    q.reduction_level.replace("_minus_", "-").replace("_percent", "%"),
                    f"{q.mean_ari:.4f}",
                    f"{q.mean_delta_ari:+.4f}",
                    f"{q.p_raw:.6g}",
                    f"{q.p_holm_60:.6g}",
                ]
            )
        text += [
            f"## {scope.title()}: largest mean ARI among the reduced conditions\n",
            "Each entry reports one fixed reducer and dimension rule across all units. These maxima are descriptive summaries; the tables report every condition.\n",
            markdown(
                [
                    "Clusterer",
                    "No DR",
                    "Reducer",
                    "Dimension",
                    "Reduced ARI",
                    "Change",
                    "Raw p",
                    "Holm p",
                ],
                rows,
            ),
        ]
    manifest = a.read(output / "tables/table_1.csv")
    text += [
        "## Table 1: Real-world datasets\n",
        markdown(
            ["Dataset", "Objects", "Features", "Clusters"],
            manifest[
                ["display_name", "n_samples", "n_features", "n_clusters"]
            ].values.tolist(),
        ),
    ]
    for i, c in enumerate(a.CS, 2):
        f = a.read(output / f"tables/table_{i}.csv")
        rows = []
        for cond, label in zip(a.CONDS[1:], conditions[1:]):
            q = f[f.condition == cond].set_index("scope")
            rows.append(
                [
                    label,
                    f"{q.loc['synthetic', 'wins']:.1f}",
                    f"{q.loc['real', 'wins']:.1f}",
                    f"{100 * q.loc['synthetic', 'mean_delta']:+.2f}",
                    f"{100 * q.loc['real', 'mean_delta']:+.2f}",
                ]
            )
        text += [
            f"## Table {i}: {plain(c)} improvements\n",
            "Wins are percentages of datasets. Average changes are 100 times the absolute ARI difference, in percentage points. Ties remain in the denominator.\n",
            markdown(
                [
                    "Condition",
                    "Synthetic wins (%)",
                    "Real wins (%)",
                    "Synthetic change (pp)",
                    "Real change (pp)",
                ],
                rows,
            ),
        ]
    for i, family in enumerate(("Circles", "Moons", "RSG", "Repliclust"), 1):
        f = a.read(output / f"tables/A{i}.csv")
        rows = []
        for c in a.CS:
            q = f[f.clusterer == c].set_index("condition").reindex(a.CONDS)
            maximum = q["mean"].max()
            values = []
            for r in q.itertuples():
                v = f"{r.mean:.3f} ± {r.std:.3f}"
                values.append(
                    "**" + v + "**"
                    if np.isclose(r.mean, maximum, atol=1e-12, rtol=0)
                    else v
                )
            rows.append([plain(c)] + values)
        text += [
            f"## Table A{i}: {family} ARI\n",
            "Mean ± sample standard deviation across datasets.\n",
            markdown(["Clusterer"] + conditions, rows),
        ]
    for i, c in enumerate(a.CS, 5):
        f = a.read(output / f"tables/A{i}.csv")
        rows = []
        for _, r in f.iterrows():
            values = r[a.CONDS].astype(float)
            rows.append(
                [r.display_name]
                + [
                    ("**" + f"{v:.3f}" + "**")
                    if np.isclose(v, values.max(), atol=1e-12, rtol=0)
                    else f"{v:.3f}"
                    for v in values
                ]
            )
        text += [
            f"## Table A{i}: {plain(c)} real-data ARI\n",
            "Each cell is the mean of ten algorithm seeds.\n",
            markdown(["Dataset"] + conditions, rows),
        ]
    for i, label in (
        (9, "Synthetic data, 45 configuration means, two-sided"),
        (10, "Real data, 20 dataset means, one-sided improvement"),
    ):
        rows = []
        for c in a.CS:
            q = (
                stats[i][stats[i].clusterer == c]
                .set_index("condition")
                .reindex(a.CONDS[1:])
            )
            rows.append([plain(c)] + [signed_p(r) for r in q.itertuples()])
        text += [
            f"## Table A{i}: {label}\n",
            "Wilcoxon signed-rank tests with Pratt zero handling. The sign gives the direction of the mean ARI change. Values below 0.001 are displayed as 0.001, following the manuscript; see the CSV for exact p-values.\n",
            markdown(["Clusterer"] + conditions[1:], rows),
            f"Nominally significant: {int((stats[i].p_raw < 0.05).sum())}/60. Significant after Holm correction across 60 tests: {int((stats[i].p_holm_60 < 0.05).sum())}/60.\n",
        ]
    for i, order, title in (
        (11, ["None"] + a.RS, "Pairwise reducer comparisons"),
        (12, a.CS, "Pairwise clusterer comparisons"),
    ):
        rows = []
        for j, name in enumerate(order):
            row = [plain(name)]
            for k, other in enumerate(order):
                if k >= j:
                    row.append("-")
                else:
                    q = (
                        stats[i][(stats[i].row == other) & (stats[i].column == name)]
                        .iloc[0]
                        .copy()
                    )
                    q["mean_delta_ari"] *= -1
                    row.append(signed_p(q))
            rows.append(row)
        text += [
            f"## Table A{i}: {title}\n",
            "Two-sided paired Wilcoxon tests on 20 real datasets. Signs describe the lower-triangle row minus column difference.\n",
            markdown([""] + [plain(x) for x in order], rows),
        ]
    text += [
        "## Figures\n",
        "Figure 1 is the bundled illustration. Figures 2-10 and A.1 are generated from the ARI records.\n",
        "[Figure 1: Synthetic datasets](figures/figure_1.pdf)\n",
    ]
    for i in range(2, 12):
        text.append(
            f"### Figure {'A.1' if i == 11 else i}\n\n![Figure {'A.1' if i == 11 else i}](figures/figure_{i}.png)\n"
        )
    (output / "README.md").write_text("\n".join(text), encoding="utf-8")


def build_report(raw_path, output):
    output = Path(output)
    for folder in ("figures", "tables"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    u = a.units(a.read(raw_path))
    a.save(u, output / "tables/dataset_mean_ari.csv.gz")
    a.save(a.configurations(u), output / "tables/synthetic_configuration_mean_ari.csv")
    _, _, stats = artifacts(u, output)
    write_results(u, output, stats)
    print("Generated Tables 1-5, A1-A12, and Figures 1-10 and A.1.")
    return u, stats
