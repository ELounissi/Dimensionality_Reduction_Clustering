from __future__ import annotations
import hashlib
import math

ALGORITHM_SEEDS = (11, 29, 47, 71, 101, 131, 173, 211, 257, 307)
STUDENT_FIGSHARE_ARTICLES = {
    "circles": 31387645,
    "moons": 31387609,
    "repliclust": 31387732,
}
REDUCERS = ("PCA", "KernelPCA", "VAE", "Isomap", "MDS")
CLUSTERERS = ("KMeans", "AHC", "GMM", "OPTICS")
LEVELS = ("k_minus_1", "25_percent", "50_percent")
MINIMUM_TARGET_DIMENSION = 2
KPCA_KERNEL = "rbf"
KPCA_GAMMA_FACTOR = 1.0
ISOMAP_DEFAULT_NEIGHBORS = 5
MDS_MAX_ITER = 300
MDS_EPS = 1e-06
MDS_RANDOM_STATE = 10
VAE_HIDDEN = (64, 32)
KMEANS_N_INIT = 100
GMM_N_INIT = 10
AHC_SELECTION_CRITERION = "silhouette"
AHC_CANDIDATE_GRID = tuple(
    (
        (linkage, metric)
        for linkage in ("ward", "complete", "average", "single")
        for metric in ("euclidean", "manhattan", "cosine")
        if not (linkage == "ward" and metric != "euclidean")
    )
)
GMM_CANDIDATE_GRID = ("full", "tied", "diag", "spherical")
OPTICS_MIN_CLUSTER_FLOOR = 5
OPTICS_ASSIGN_NOISE = True
REAL_FIGSHARE_ARTICLE = 31386421
RSG_FIGSHARE_ARTICLE = 31383301
SPECTF_URL = "https://archive.ics.uci.edu/static/public/96/spectf+heart.zip"


def stable_seed(text: str, offset: int = 0) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "little") + offset) % (2**32 - 1)


def target_dimensions(
    n_clusters: int, n_features: int, n_samples: int
) -> dict[str, int]:
    if n_features < 2 or n_samples < 2:
        raise ValueError("at least two features and samples are required")
    maximum = min(n_features - 1, n_samples - 1)
    return {
        "k_minus_1": min(max(n_clusters - 1, MINIMUM_TARGET_DIMENSION), maximum),
        "25_percent": min(max(1, math.ceil(0.25 * n_features)), maximum),
        "50_percent": min(max(1, math.ceil(0.5 * n_features)), maximum),
    }


def algorithm_seeds(scope: str, dataset_id: str) -> tuple[int, ...]:
    if scope == "real":
        return ALGORITHM_SEEDS
    return (stable_seed(dataset_id),)


def optics_setting_name(min_samples, xi, fraction):
    return f"ms{min_samples}/xi{xi:g}/mc{fraction:g}"


def optics_candidates():
    return tuple(
        (
            (optics_setting_name(ms, 0.05, fraction), ms, 0.05, fraction)
            for ms in range(5, 11)
            for fraction in (round(0.05 * step, 2) for step in range(1, 20))
        )
    )
