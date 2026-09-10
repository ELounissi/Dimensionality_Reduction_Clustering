"""Integrity and numerical checks for the bundled benchmark records."""

import hashlib
import json

import numpy as np
import pandas as pd

from . import analysis as a

KEY = a.KEY + ["clusterer"]


def verify(root):
    manifest = json.loads((root / "frozen/checksums.json").read_text())
    for name, expected in manifest.items():
        observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    raw = a.read(root / "frozen/raw_results.csv.gz")
    if (
        len(raw) != 87360
        or raw.dataset_id.nunique() != 1185
        or raw.duplicated(KEY).any()
    ):
        raise ValueError("Incomplete frozen results")
    chunks = []
    for frame in pd.read_csv(
        root / "frozen/candidates.csv.gz",
        keep_default_na=False,
        chunksize=200000,
        float_precision="round_trip",
    ):
        frame = frame.sort_values(
            ["criterion", "setting"], ascending=[False, True], kind="stable"
        )
        chunks.append(frame.groupby(KEY, sort=False).head(1))
    best = (
        pd.concat(chunks, ignore_index=True)
        .sort_values(["criterion", "setting"], ascending=[False, True], kind="stable")
        .groupby(KEY, sort=False)
        .head(1)
    )
    compared = raw.merge(
        best, on=KEY, suffixes=("_reported", "_computed"), validate="one_to_one"
    )
    if (
        len(compared) != len(raw)
        or not compared.setting_reported.eq(compared.setting_computed).all()
    ):
        raise ValueError("A reported setting differs from its internal-score optimum")
    if not np.allclose(
        compared.ari_reported, compared.ari_computed, rtol=0, atol=1e-12
    ):
        raise ValueError("A reported ARI differs from its recorded fit")
    u = a.units(raw)
    real, synthetic = a.versus_baseline(u, "real"), a.versus_baseline(u, "synthetic")
    if (
        u[u.scope == "real"].seeds.ne(10).any()
        or u[u.scope == "synthetic"].seeds.ne(1).any()
    ):
        raise ValueError("Incorrect repeat counts")
    expected = json.loads((root / "frozen/statistics_reference.json").read_text())
    for name, table in (
        ("A9", synthetic),
        ("A10", real),
        ("A11", a.pairwise(u, "reducer")),
        ("A12", a.pairwise(u, "clusterer")),
    ):
        if not np.allclose(table.p_raw, expected[name], rtol=0, atol=1e-12):
            raise ValueError(f"Statistical disagreement: {name}")
    print(f"Verified {len(raw):,} results, 1,185 datasets, and all A9-A12 p-values.")
    return raw
