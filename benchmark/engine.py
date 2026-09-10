from __future__ import annotations
import random
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score
from .config import CLUSTERERS, algorithm_seeds, stable_seed, target_dimensions
from .datasets import DatasetSpec, load_dataset
from .methods import cluster_all, reduce_cpu, vae_embedding

CONDITION_COLUMNS = ["dataset_id", "algorithm_seed", "reducer", "reduction_level"]


def _condition_row(
    spec: DatasetSpec,
    x: np.ndarray,
    seed: int,
    reducer: str,
    level: str,
    components: int,
    metadata: dict,
    error: Exception | None = None,
) -> dict:
    return {
        **asdict(spec),
        "algorithm_seed": int(seed),
        "embedding_seed": int(seed)
        if reducer == "VAE"
        else stable_seed(spec.dataset_id),
        "n_samples": int(len(x)),
        "reducer": reducer,
        "reduction_level": level,
        "n_components": int(components),
        **metadata,
        "status": "ok" if error is None else "error",
        "error": "" if error is None else repr(error),
    }


def _candidate_rows(
    spec: DatasetSpec,
    y: np.ndarray,
    embedding: np.ndarray,
    reducer: str,
    level: str,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    for clusterer in CLUSTERERS:
        started = time.perf_counter()
        try:
            candidates = cluster_all(embedding, clusterer, spec.n_clusters, seed)
        except Exception as exc:
            rows.append(
                {
                    "dataset_id": spec.dataset_id,
                    "algorithm_seed": int(seed),
                    "reducer": reducer,
                    "reduction_level": level,
                    "clusterer": clusterer,
                    "setting": "",
                    "ari": np.nan,
                    "criterion": np.nan,
                    "n_predicted_clusters": 0,
                    "noise_fraction": np.nan,
                    "clustering_seconds": time.perf_counter() - started,
                    "status": "error",
                    "error": repr(exc),
                }
            )
            continue
        elapsed = (time.perf_counter() - started) / len(candidates)
        for setting, predicted, criterion in candidates:
            ari = float(adjusted_rand_score(y, predicted))
            if not -0.5 <= ari <= 1.0:
                raise ValueError(f"ARI outside mathematical bounds: {ari}")
            rows.append(
                {
                    "dataset_id": spec.dataset_id,
                    "algorithm_seed": int(seed),
                    "reducer": reducer,
                    "reduction_level": level,
                    "clusterer": clusterer,
                    "setting": setting,
                    "ari": ari,
                    "criterion": float(criterion),
                    "n_predicted_clusters": int(
                        len(set(predicted.tolist())) - (1 if -1 in predicted else 0)
                    ),
                    "noise_fraction": float(np.mean(predicted == -1)),
                    "clustering_seconds": elapsed,
                    "status": "ok",
                    "error": "",
                }
            )
    return rows


def dataset_task(spec: DatasetSpec, data_root: Path) -> tuple[list[dict], list[dict]]:
    task_seed = stable_seed(spec.dataset_id)
    random.seed(task_seed)
    np.random.seed(task_seed)
    x, y = load_dataset(spec, data_root)
    seeds = algorithm_seeds(spec.scope, spec.dataset_id)
    candidates: list[dict] = []
    conditions: list[dict] = []
    for seed in seeds:
        conditions.append(
            _condition_row(
                spec,
                x,
                seed,
                "None",
                "original",
                x.shape[1],
                {"embedding_seconds": 0.0},
            )
        )
        candidates.extend(_candidate_rows(spec, y, x, "None", "original", seed))
    dimensions = target_dimensions(spec.n_clusters, spec.n_features, len(x))
    for reducer in ("PCA", "KernelPCA", "Isomap", "MDS"):
        for level, components in dimensions.items():
            try:
                embedding, metadata = reduce_cpu(x, reducer, components, task_seed)
            except Exception as exc:
                for seed in seeds:
                    conditions.append(
                        _condition_row(
                            spec,
                            x,
                            seed,
                            reducer,
                            level,
                            components,
                            {"embedding_seconds": 0.0},
                            exc,
                        )
                    )
                continue
            for seed in seeds:
                conditions.append(
                    _condition_row(spec, x, seed, reducer, level, components, metadata)
                )
                candidates.extend(
                    _candidate_rows(spec, y, embedding, reducer, level, seed)
                )
    for seed in seeds:
        for level, components in dimensions.items():
            try:
                embedding, metadata = vae_embedding(x, components, seed)
            except Exception as exc:
                conditions.append(
                    _condition_row(
                        spec,
                        x,
                        seed,
                        "VAE",
                        level,
                        components,
                        {"embedding_seconds": 0.0},
                        exc,
                    )
                )
                continue
            conditions.append(
                _condition_row(spec, x, seed, "VAE", level, components, metadata)
            )
            candidates.extend(_candidate_rows(spec, y, embedding, "VAE", level, seed))
    return (candidates, conditions)
