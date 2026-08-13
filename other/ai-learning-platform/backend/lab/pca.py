"""
PCA & eigenvectors: correlated 2D data projected onto its principal components.

Uses numpy eigendecomposition of the covariance matrix (so the geometry of
eigenvectors is visible) plus an optional sklearn PCA for the projection.
"""

from typing import Any
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA


def compute(params: dict[str, Any]) -> dict[str, Any]:
    n = int(params.get("samples", 150))
    corr = float(params.get("correlation", 0.75))
    spread = float(params.get("spread", 1.6))
    components = int(params.get("components", 1))
    seed = int(params.get("seed", 3))

    rng = np.random.default_rng(seed)
    # Build a covariance matrix with the requested correlation and spread ratio
    # between the two axes.
    var1, var2 = spread, 1.0 / spread
    cov = np.array([
        [var1, corr * np.sqrt(var1 * var2)],
        [corr * np.sqrt(var1 * var2), var2],
    ])
    mean = np.array([0.0, 0.0])
    X = rng.multivariate_normal(mean, cov, size=n)

    # Eigen-decomposition of the (sample) covariance matrix.
    centered = X - X.mean(axis=0)
    sample_cov = (centered.T @ centered) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(sample_cov)
    # eigh returns ascending; sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Project via sklearn PCA (keeps the sklearn dependency honest) and
    # reconstruct with the requested number of components.
    pca = SklearnPCA(n_components=2)
    pca.fit(X)
    comps = pca.components_
    explained = pca.explained_variance_ratio_

    reduced = X @ comps[:components].T
    if components == 0:
        reconstructed = np.tile(X.mean(axis=0), (n, 1))
    else:
        reconstructed = reduced @ comps[:components] + X.mean(axis=0)

    # Residuals as line segments original -> reconstructed
    residuals = []
    for a, b in zip(X, reconstructed):
        residuals.append([
            [round(float(a[0]), 4), round(float(a[1]), 4)],
            [round(float(b[0]), 4), round(float(b[1]), 4)],
        ])

    total_var = float(np.sum(explained))
    kept_var = float(np.sum(explained[:components])) if components > 0 else 0.0

    # Axis bounds with a little padding
    all_pts = np.vstack([X, reconstructed])
    lo = float(all_pts.min() - 0.5)
    hi = float(all_pts.max() + 0.5)

    def vec(v: np.ndarray, length: float) -> list[list[float]]:
        unit = v / (np.linalg.norm(v) + 1e-9)
        return [[0.0, 0.0], [float(unit[0] * length), float(unit[1] * length)]]

    return {
        "domain": {"x": [lo, hi], "y": [lo, hi]},
        "points": np.round(X, 4).tolist(),
        "reconstructed": np.round(reconstructed, 4).tolist(),
        "residuals": residuals[:200],
        "mean": [round(float(X.mean(axis=0)[0]), 4), round(float(X.mean(axis=0)[1]), 4)],
        "eigenvalues": [round(float(e), 4) for e in eigvals],
        "eigenvectors": [
            {"name": f"PC{i+1}", "segment": vec(eigvecs[:, i], float(np.sqrt(eigvals[i])) * 2.0),
             "variance": round(float(explained[i]), 4)}
            for i in range(2)
        ],
        "components": components,
        "explainedVariance": [round(float(v), 4) for v in explained],
        "keptVariance": round(kept_var / total_var if total_var else 0.0, 4),
    }
