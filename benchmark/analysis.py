import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

CS = ["KMeans", "AHC", "GMM", "OPTICS"]
RS = ["PCA", "KernelPCA", "VAE", "Isomap", "MDS"]
LS = ["k_minus_1", "25_percent", "50_percent"]
CONDS = ["None__original"] + [r + "__" + l for r in RS for l in LS]
KEY = ["dataset_id", "algorithm_seed", "reducer", "reduction_level"]
GROUP = ["data_type", "clusterer", "reducer", "reduction_level"]


def read(p, **kw):
    return pd.read_csv(p, keep_default_na=False, float_precision="round_trip", **kw)


def save(f, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(p, index=False, float_format="%.17g")


def holm(p):
    p = np.asarray(p)
    ix = np.argsort(p)
    z = np.empty(len(p))
    z[ix] = np.minimum(1, np.maximum.accumulate(p[ix] * (len(p) - np.arange(len(p)))))
    return z


def test(d, alternative):
    d = np.asarray(d, float)
    assert np.isfinite(d).all()
    if np.all(d == 0):
        statistic, p = (0.0, 1.0)
    else:
        result = wilcoxon(
            d, alternative=alternative, zero_method="pratt", method="auto"
        )
        statistic, p = (float(result.statistic), float(result.pvalue))
    r = rankdata(abs(d[d != 0])) if np.any(d != 0) else np.array([])
    dn = d[d != 0]
    rb = float((r[dn > 0].sum() - r[dn < 0].sum()) / r.sum()) if len(r) else 0.0
    return dict(
        n_units=len(d),
        n_nonzero=int((d != 0).sum()),
        wins=int((d > 0).sum()),
        ties=int((d == 0).sum()),
        losses=int((d < 0).sum()),
        mean_delta_ari=float(d.mean()),
        median_delta_ari=float(np.median(d)),
        rank_biserial=rb,
        statistic=statistic,
        p_raw=p,
    )


def units(raw):
    cols = [
        "dataset_id",
        "display_name",
        "scope",
        "family",
        "n_clusters",
        "n_features",
        "n_samples",
        "clusterer",
        "reducer",
        "reduction_level",
    ]
    u = raw.groupby(cols, as_index=False).agg(
        ari=("ari", "mean"), seeds=("algorithm_seed", "nunique")
    )
    u["condition"] = u.reducer + "__" + u.reduction_level
    return u


def configurations(u):
    s = (
        u[u.scope == "synthetic"]
        .groupby(
            [
                "family",
                "n_clusters",
                "n_features",
                "n_samples",
                "clusterer",
                "reducer",
                "reduction_level",
                "condition",
            ],
            as_index=False,
        )
        .agg(ari=("ari", "mean"), replicates=("dataset_id", "nunique"))
    )
    s["unit"] = (
        s[["family", "n_clusters", "n_features", "n_samples"]]
        .astype(str)
        .agg("|".join, axis=1)
    )
    assert s.unit.nunique() == 45
    return s


def versus_baseline(u, scope):
    f = (
        u[u.scope == "real"].assign(unit=lambda x: x.dataset_id)
        if scope == "real"
        else configurations(u)
    )
    rows = []
    for c in CS:
        w = (
            f[f.clusterer == c]
            .pivot(index="unit", columns="condition", values="ari")
            .reindex(columns=CONDS)
        )
        assert not w.isna().any().any() and len(w) == (20 if scope == "real" else 45)
        for cond in CONDS[1:]:
            rows.append(
                dict(
                    clusterer=c,
                    condition=cond,
                    reducer=cond.split("__")[0],
                    reduction_level=cond.split("__")[1],
                    mean_ari=float(w[cond].mean()),
                    baseline_ari=float(w[CONDS[0]].mean()),
                    **test(
                        w[cond] - w[CONDS[0]],
                        "greater" if scope == "real" else "two-sided",
                    ),
                )
            )
    result = pd.DataFrame(rows)
    result["p_holm_60"] = holm(result.p_raw)
    for _, ix in result.groupby("clusterer").groups.items():
        result.loc[ix, "p_holm_15"] = holm(result.loc[ix, "p_raw"])
    return result


def pairwise(u, kind):
    real = u[u.scope == "real"]
    order = ["None"] + RS if kind == "reducer" else CS
    w = real.groupby(["dataset_id", kind]).ari.mean().unstack(kind)[order]
    rows = []
    for a, b in itertools.combinations(order, 2):
        rows.append(
            dict(
                row=a,
                column=b,
                mean_row=float(w[a].mean()),
                mean_column=float(w[b].mean()),
                **test(w[a] - w[b], "two-sided"),
            )
        )
    f = pd.DataFrame(rows)
    f["p_holm"] = holm(f.p_raw)
    return f
