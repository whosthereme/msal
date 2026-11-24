from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

PCA_COMPONENTS = 64


@dataclass
class DiscoveredCells:

    labels: np.ndarray
    cluster_count: int
    seed: int
    adjusted_rand_index: float
    cluster_purity: dict[int, dict[str, float]]
    mean_purity: float
    explained_variance_ratio: float
    pca_components: int

    def as_report(self) -> dict:
        return {
            "cluster_count": self.cluster_count,
            "seed": self.seed,
            "adjusted_rand_index": self.adjusted_rand_index,
            "mean_purity": self.mean_purity,
            "cluster_purity": {str(key): value for key, value in self.cluster_purity.items()},
            "pca_explained_variance_ratio": self.explained_variance_ratio,
            "pca_components": self.pca_components,
            "pca_components_requested": PCA_COMPONENTS,
        }


def _purity(
    labels: np.ndarray, phase_indices: np.ndarray, phase_names: tuple[str, ...]
) -> tuple[dict[int, dict[str, float]], float]:
    table: dict[int, dict[str, float]] = {}
    weighted = 0.0
    for cluster in np.unique(labels):
        selected = labels == cluster
        counts = np.bincount(phase_indices[selected], minlength=len(phase_names))
        dominant = int(counts.argmax())
        share = float(counts[dominant] / max(counts.sum(), 1))
        table[int(cluster)] = {
            "rows": int(selected.sum()),
            "dominant_phase": phase_names[dominant],
            "dominant_phase_share": share,
        }
        weighted += share * float(selected.sum())
    return table, float(weighted / max(len(labels), 1))


def discover_cells(
    state_features: np.ndarray,
    fit_rows: np.ndarray,
    all_rows: np.ndarray,
    phase_indices: np.ndarray,
    phase_names: tuple[str, ...],
    *,
    cluster_count: int,
    seed: int,
) -> tuple[DiscoveredCells, np.ndarray]:
    if cluster_count < 2:
        raise ValueError("cluster_count must be >= 2")
    fit_block = np.asarray(state_features[fit_rows], dtype=np.float32)
    components = min(PCA_COMPONENTS, min(fit_block.shape))
    projector = PCA(n_components=components, random_state=seed)
    reduced_fit = projector.fit_transform(fit_block)
    model = KMeans(n_clusters=cluster_count, random_state=seed, n_init=10)
    model.fit(reduced_fit)

    fit_labels = model.predict(reduced_fit)
    purity, mean_purity = _purity(fit_labels, phase_indices[fit_rows], phase_names)
    scored = DiscoveredCells(
        labels=fit_labels,
        cluster_count=cluster_count,
        seed=seed,
        adjusted_rand_index=float(
            adjusted_rand_score(phase_indices[fit_rows], fit_labels)
        ),
        cluster_purity=purity,
        mean_purity=mean_purity,
        explained_variance_ratio=float(projector.explained_variance_ratio_.sum()),
        pca_components=int(components),
    )

    labels = np.empty(len(all_rows), dtype=np.int64)
    for start in range(0, len(all_rows), 4096):
        subset = all_rows[start : start + 4096]
        block = np.asarray(state_features[subset], dtype=np.float32)
        labels[start : start + len(subset)] = model.predict(projector.transform(block))
    return scored, labels
