"""Download the benchmark data, fit every representation, and record the results."""

import concurrent.futures
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .config import algorithm_seeds
from .datasets import dataset_specs, fetch_data
from .engine import CONDITION_COLUMNS, dataset_task

KEY = CONDITION_COLUMNS + ["clusterer"]


def fitted_results(candidates, representations):
    if candidates.duplicated(KEY + ["setting"]).any():
        raise ValueError("Duplicate candidate records")
    if representations.duplicated(CONDITION_COLUMNS).any():
        raise ValueError("Duplicate representation records")
    for frame in (candidates, representations):
        if not frame.status.eq("ok").all():
            raise ValueError("Failed fits must be resolved before reporting")
    if not np.isfinite(candidates.ari).all():
        raise ValueError("Non-finite ARI")
    ordered = candidates.sort_values(
        ["criterion", "setting"], ascending=[False, True], kind="stable"
    )
    fits = ordered.groupby(KEY, sort=False).head(1)
    columns = CONDITION_COLUMNS + [
        "display_name",
        "scope",
        "family",
        "n_clusters",
        "n_features",
        "n_samples",
        "replicate",
        "source",
        "n_components",
    ]
    result = fits.merge(
        representations[columns], on=CONDITION_COLUMNS, validate="many_to_one"
    )
    return result.sort_values(KEY, kind="stable")


def _one(spec, data_root, output):
    candidates, representations = map(pd.DataFrame, dataset_task(spec, data_root))
    result = fitted_results(candidates, representations)
    expected = 64 * len(algorithm_seeds(spec.scope, spec.dataset_id))
    if len(result) != expected or len(representations) != expected // 4:
        raise ValueError(f"Incomplete dataset: {spec.dataset_id}")
    folder = output / "datasets" / spec.dataset_id
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("candidates", candidates),
        ("representations", representations),
        ("raw_results", result),
    ):
        frame.to_csv(folder / f"{name}.csv.gz", index=False, float_format="%.17g")
    (folder / "complete.json").write_text(
        json.dumps({"rows": len(result)}) + "\n", encoding="utf-8"
    )
    return spec.dataset_id


def run(data_root, output, workers):
    required = {
        "numpy": "2.2.2",
        "pandas": "2.2.3",
        "scipy": "1.15.1",
        "scikit-learn": "1.7.1",
        "torch": "2.9.1",
    }
    mismatch = [
        name
        for name, version in required.items()
        if importlib.metadata.version(name).split("+")[0] != version
    ]
    if platform.python_version_tuple()[:2] != ("3", "12") or mismatch:
        raise RuntimeError(
            "Full fitting requires Python 3.12 and the versions in requirements.txt."
        )
    fetch_data(data_root)
    specs = dataset_specs(data_root)
    if len(specs) != 1185 or len({s.dataset_id for s in specs}) != 1185:
        raise ValueError("Expected 1,185 distinct datasets")
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source.update(path.name.encode())
        source.update(path.read_bytes())
    signature = {
        "source_sha256": source.hexdigest(),
        "data_sha256": hashlib.sha256(
            (data_root / "source_manifest.json").read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "packages": {
            n: importlib.metadata.version(n)
            for n in ("numpy", "pandas", "scipy", "scikit-learn", "torch")
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "run.json"
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError(
            "Existing run has different code, data, or dependencies. Use a new output directory."
        )
    manifest.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    pending = [
        s
        for s in specs
        if not (output / "datasets" / s.dataset_id / "complete.json").exists()
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        jobs = {
            executor.submit(_one, spec, data_root, output): spec for spec in pending
        }
        for count, future in enumerate(
            concurrent.futures.as_completed(jobs), len(specs) - len(pending) + 1
        ):
            name = future.result()
            print(f"Completed {count}/{len(specs)}: {name}", flush=True)
    raw = pd.concat(
        [
            pd.read_csv(
                output / "datasets" / s.dataset_id / "raw_results.csv.gz",
                keep_default_na=False,
            )
            for s in specs
        ],
        ignore_index=True,
    )
    if len(raw) != 87360 or raw.duplicated(KEY).any():
        raise ValueError("Incomplete benchmark")
    path = output / "raw_results.csv.gz"
    raw.to_csv(path, index=False, float_format="%.17g")
    return path
