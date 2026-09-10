import math
import os
import random
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.cluster import OPTICS, AgglomerativeClustering, KMeans, cluster_optics_xi
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS, Isomap
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from .config import (
    AHC_CANDIDATE_GRID,
    AHC_SELECTION_CRITERION,
    GMM_CANDIDATE_GRID,
    GMM_N_INIT,
    ISOMAP_DEFAULT_NEIGHBORS,
    KMEANS_N_INIT,
    KPCA_GAMMA_FACTOR,
    KPCA_KERNEL,
    MDS_EPS,
    MDS_MAX_ITER,
    MDS_RANDOM_STATE,
    OPTICS_MIN_CLUSTER_FLOOR,
    optics_candidates,
    OPTICS_ASSIGN_NOISE,
    VAE_HIDDEN,
)


def _vae_widths(
    n_components: int, hidden: tuple[int, ...] = VAE_HIDDEN
) -> tuple[int, ...]:
    return tuple((max(n_components, width) for width in hidden))


def _connected_neighbor_count(x: np.ndarray, minimum_neighbors: int) -> tuple[int, int]:
    maximum = len(x) - 1
    neighbors = min(maximum, minimum_neighbors)
    initial = neighbors
    while True:
        graph = kneighbors_graph(
            x, n_neighbors=neighbors, mode="connectivity", include_self=True, n_jobs=1
        )
        components = int(
            connected_components(
                graph.maximum(graph.T), directed=False, return_labels=False
            )
        )
        if components == 1 or neighbors == maximum:
            return (neighbors, initial)
        neighbors = min(maximum, 2 * neighbors)


def _isomap(x: np.ndarray, n_components: int) -> tuple[np.ndarray, dict]:
    return _isomap_with_neighbors(x, n_components, ISOMAP_DEFAULT_NEIGHBORS)


def _isomap_with_neighbors(
    x: np.ndarray, n_components: int, minimum_neighbors: int
) -> tuple[np.ndarray, dict]:
    neighbors, initial_neighbors = _connected_neighbor_count(x, minimum_neighbors)
    model = Isomap(
        n_components=1, n_neighbors=neighbors, eigen_solver="arpack", n_jobs=1
    )
    model.fit(x)
    squared = np.square(np.asarray(model.dist_matrix_, dtype=np.float64))
    row_mean = squared.mean(axis=1)
    grand_mean = float(squared.mean())
    gram = -0.5 * (squared - row_mean[:, None] - row_mean[None, :] + grand_mean)
    if len(x) <= 300 or n_components >= len(x) - 1:
        eigenvalues, eigenvectors = eigh(
            gram,
            subset_by_index=(len(x) - n_components, len(x) - 1),
            check_finite=False,
            driver="evr",
        )
    else:
        initial = np.linspace(-1.0, 1.0, len(x))
        initial /= np.linalg.norm(initial)
        eigenvalues, eigenvectors = eigsh(
            gram, k=n_components, which="LA", v0=initial, tol=1e-10
        )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    tolerance = max(np.finfo(float).eps, float(eigenvalues[0]) * 1e-12)
    positive = eigenvalues > tolerance
    actual = int(positive.sum())
    if actual == 0:
        raise ValueError("Isomap geodesic matrix has no positive eigen-directions")
    embedding = np.zeros((len(x), n_components), dtype=np.float64)
    embedding[:, :actual] = eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])
    return (
        embedding,
        {
            "isomap_neighbors": neighbors,
            "isomap_initial_neighbors": initial_neighbors,
            "isomap_positive_dimensions": actual,
            "isomap_zero_padded_dimensions": n_components - actual,
        },
    )


def _pca_scores(x: np.ndarray, n_components: int) -> tuple[np.ndarray, str]:
    try:
        return (
            PCA(n_components=n_components, svd_solver="full").fit_transform(x),
            "full",
        )
    except np.linalg.LinAlgError:
        return (
            PCA(n_components=n_components, svd_solver="covariance_eigh").fit_transform(
                x
            ),
            "covariance_eigh",
        )


def _kernel_pca(x, n_components, seed):
    gamma = KPCA_GAMMA_FACTOR / x.shape[1]
    model = KernelPCA(
        n_components=n_components,
        kernel=KPCA_KERNEL,
        gamma=gamma,
        eigen_solver="arpack",
        random_state=seed,
        n_jobs=1,
    )
    return (model.fit_transform(x), {"kpca_kernel": KPCA_KERNEL, "kpca_gamma": gamma})


def _mds(x, n_components):
    classical = _pca_scores(x, n_components)[0]
    model = MDS(
        n_components=n_components,
        metric=True,
        n_init=1,
        max_iter=MDS_MAX_ITER,
        eps=MDS_EPS,
        random_state=MDS_RANDOM_STATE,
        dissimilarity="euclidean",
        normalized_stress=False,
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", UserWarning)
        embedding = model.fit_transform(x, init=classical)
    return (
        embedding,
        {
            "mds_variant": "smacof_metric_classical_init",
            "mds_n_init": 1,
            "mds_max_iter": MDS_MAX_ITER,
            "mds_stress": float(model.stress_),
            "mds_n_iter": int(model.n_iter_),
        },
    )


def reduce_cpu(
    x: np.ndarray, method: str, n_components: int, seed: int
) -> tuple[np.ndarray, dict]:
    started = time.perf_counter()
    metadata: dict[str, float | int | str] = {}
    if method == "PCA":
        embedding, solver = _pca_scores(x, n_components)
        metadata = {"pca_solver": solver}
    elif method == "KernelPCA":
        embedding, metadata = _kernel_pca(x, n_components, seed)
    elif method == "Isomap":
        embedding, metadata = _isomap(x, n_components)
    elif method == "MDS":
        embedding, metadata = _mds(x, n_components)
    else:
        raise ValueError(method)
    embedding = np.asarray(embedding, dtype=np.float64)
    if embedding.shape != (len(x), n_components) or not np.isfinite(embedding).all():
        raise ValueError(
            f"{method} produced an invalid embedding of shape {embedding.shape}"
        )
    return (embedding, {"embedding_seconds": time.perf_counter() - started, **metadata})


def _internal_score(x, labels, criterion):
    if len(set(labels.tolist())) < 2:
        return -np.inf
    return float(silhouette_score(x, labels))


def cluster_all(
    x: np.ndarray, method: str, n_clusters: int, seed: int
) -> list[tuple[str, np.ndarray, float]]:
    results: list[tuple[str, np.ndarray, float]] = []
    if method == "KMeans":
        labels = np.asarray(
            KMeans(
                n_clusters=n_clusters,
                init="k-means++",
                n_init=KMEANS_N_INIT,
                max_iter=300,
                random_state=seed,
            ).fit_predict(x),
            dtype=int,
        )
        results.append(
            ("n_init=100", labels, _internal_score(x, labels, AHC_SELECTION_CRITERION))
        )
    elif method == "AHC":
        for linkage, metric in AHC_CANDIDATE_GRID:
            try:
                labels = np.asarray(
                    AgglomerativeClustering(
                        n_clusters=n_clusters, linkage=linkage, metric=metric
                    ).fit_predict(x),
                    dtype=int,
                )
            except ValueError:
                continue
            results.append(
                (
                    f"{linkage}/{metric}",
                    labels,
                    _internal_score(x, labels, AHC_SELECTION_CRITERION),
                )
            )
    elif method == "GMM":
        for covariance in GMM_CANDIDATE_GRID:
            model = GaussianMixture(
                n_components=n_clusters,
                covariance_type=covariance,
                n_init=GMM_N_INIT,
                max_iter=300,
                reg_covar=1e-05,
                random_state=seed,
            )
            labels = np.asarray(model.fit_predict(x), dtype=int)
            results.append((covariance, labels, -float(model.bic(x))))
    elif method == "OPTICS":
        results.extend(_optics_all(x, n_clusters))
    else:
        raise ValueError(method)
    if not results:
        raise ValueError(f"no admissible candidate for {method}")
    return results


def _optics_all(x: np.ndarray, n_clusters: int) -> list[tuple[str, np.ndarray, float]]:
    grouped: dict[int, list[tuple[str, float, float]]] = {}
    for name, min_samples, xi, fraction in optics_candidates():
        grouped.setdefault(min_samples, []).append((name, xi, fraction))
    smallest = max(OPTICS_MIN_CLUSTER_FLOOR, math.ceil(0.02 * len(x)))
    results: list[tuple[str, np.ndarray, float]] = []
    for min_samples in sorted(grouped):
        if min_samples >= len(x):
            continue
        model = OPTICS(
            min_samples=min_samples,
            max_eps=np.inf,
            metric="minkowski",
            p=2,
            cluster_method="xi",
            xi=0.05,
            min_cluster_size=smallest,
            n_jobs=1,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            model.fit(x)
            for name, xi, fraction in grouped[min_samples]:
                minimum_cluster = max(
                    OPTICS_MIN_CLUSTER_FLOOR, math.ceil(fraction * len(x))
                )
                if minimum_cluster >= len(x):
                    continue
                labels, _ = cluster_optics_xi(
                    reachability=model.reachability_,
                    predecessor=model.predecessor_,
                    ordering=model.ordering_,
                    min_samples=min_samples,
                    min_cluster_size=minimum_cluster,
                    xi=xi,
                    predecessor_correction=True,
                )
                labels = np.asarray(labels, dtype=int)
                found = len(set(labels.tolist())) - (1 if -1 in labels else 0)
                noise = float(np.mean(labels == -1))
                criterion = -(1000.0 * abs(found - n_clusters) + noise)
                if OPTICS_ASSIGN_NOISE:
                    labels = assign_noise_to_nearest_cluster(x, labels)
                results.append((name, labels, criterion))
    return results


def vae_embedding(
    x: np.ndarray, n_components: int, seed: int
) -> tuple[np.ndarray, dict]:
    import torch
    import torch.nn.functional as functional

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    widths = _vae_widths(n_components)

    class VAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            encoder_layers: list[torch.nn.Module] = []
            previous = x.shape[1]
            for width in widths:
                encoder_layers.extend(
                    [torch.nn.Linear(previous, width), torch.nn.ReLU()]
                )
                previous = width
            encoder_layers.extend(
                [torch.nn.BatchNorm1d(previous), torch.nn.Dropout(0.4)]
            )
            self.encoder = torch.nn.Sequential(*encoder_layers)
            self.mean = torch.nn.Linear(previous, n_components)
            self.log_variance = torch.nn.Linear(previous, n_components)
            decoder_layers: list[torch.nn.Module] = []
            previous = n_components
            for width in reversed(widths):
                decoder_layers.extend(
                    [torch.nn.Linear(previous, width), torch.nn.ReLU()]
                )
                previous = width
            decoder_layers.extend(
                [torch.nn.BatchNorm1d(previous), torch.nn.Dropout(0.4)]
            )
            decoder_layers.append(torch.nn.Linear(previous, x.shape[1]))
            self.decoder = torch.nn.Sequential(*decoder_layers)

        def forward(self, values):
            hidden = self.encoder(values)
            mean = self.mean(hidden)
            log_variance = self.log_variance(hidden).clamp(-20.0, 10.0)
            latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
            return (self.decoder(latent), mean, log_variance)

    started = time.perf_counter()
    tensor = torch.as_tensor(x, dtype=torch.float32, device="cpu")
    generator = torch.Generator().manual_seed(seed)
    model = VAE().to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_samples = len(tensor)
    epoch_reconstruction = math.nan
    epoch_kl = math.nan
    budget = 100
    trained_epochs = 0
    for epoch in range(budget):
        model.train()
        kl_weight = 1.0
        permutation = torch.randperm(n_samples, generator=generator).to("cpu")
        total_reconstruction = 0.0
        total_kl = 0.0
        for start in range(0, n_samples, 64):
            batch = tensor[permutation[start : start + 64]]
            if len(batch) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mean, log_variance = model(batch)
            reconstruction_loss = functional.mse_loss(
                reconstruction, batch, reduction="sum"
            ) / len(batch)
            kl_loss = (
                -0.5
                * torch.sum(1.0 + log_variance - mean.square() - log_variance.exp())
                / len(batch)
            )
            loss = reconstruction_loss + kl_weight * kl_loss
            if not torch.isfinite(loss):
                raise RuntimeError("VAE training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            total_reconstruction += float(reconstruction_loss.detach()) * len(batch)
            total_kl += float(kl_loss.detach()) * len(batch)
        epoch_reconstruction = total_reconstruction / n_samples
        epoch_kl = total_kl / n_samples
        trained_epochs = epoch + 1
    model.eval()
    with torch.no_grad():
        embedding = model.mean(model.encoder(tensor)).cpu().numpy().astype(np.float64)
    if not np.isfinite(embedding).all():
        raise RuntimeError("VAE produced a non-finite embedding")
    return (
        embedding,
        {
            "embedding_seconds": time.perf_counter() - started,
            "vae_epochs": trained_epochs,
            "vae_batch_size": 64,
            "vae_beta": 1.0,
            "vae_decoder_variance": "fixed",
            "vae_warmup_epochs": 0,
            "vae_final_reconstruction": epoch_reconstruction,
            "vae_final_kl": epoch_kl,
            "vae_hidden": "-".join((str(width) for width in widths)),
            "vae_normalize": bool(True),
            "vae_learning_rate": 0.001,
            "vae_early_stopping": bool(False),
        },
    )


def assign_noise_to_nearest_cluster(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int).copy()
    noise = labels == -1
    if not noise.any():
        return labels
    clusters = sorted(set(labels[~noise].tolist()))
    if not clusters:
        return np.zeros(len(labels), dtype=int)
    centres = np.vstack([x[labels == label].mean(axis=0) for label in clusters])
    distances = ((x[noise][:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    labels[noise] = np.asarray(clusters, dtype=int)[distances.argmin(axis=1)]
    return labels
