"""
Probability distributions — compute PDFs/PMFs and sampled histograms.
The frontend only renders the returned arrays; all math runs here.
"""

from typing import Any
import numpy as np


def _gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _uniform(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.where((x >= lo) & (x <= hi), 1.0 / (hi - lo), 0.0)


def _exponential(x: np.ndarray, rate: float) -> np.ndarray:
    return np.where(x >= 0, rate * np.exp(-rate * x), 0.0)


def _laplace(x: np.ndarray, mu: float, b: float) -> np.ndarray:
    return np.exp(-np.abs(x - mu) / b) / (2 * b)


def _binomial(k: np.ndarray, n: int, p: float) -> np.ndarray:
    from math import comb

    return np.array([comb(n, int(ki)) * p**ki * (1 - p) ** (n - ki) for ki in k])


def _poisson(k: np.ndarray, rate: float) -> np.ndarray:
    from math import exp, lgamma

    return np.array(
        [float(np.exp(-rate + ki * np.log(rate) - lgamma(ki + 1))) for ki in k]
    )


DISTRIBUTIONS: dict[str, dict[str, Any]] = {
    "gaussian": {
        "fn": _gaussian,
        "name": "Gaussian",
        "formula": "N(mu, sigma^2)",
        "discrete": False,
        "params": [("mu", -3.0, 3.0, 0.0), ("sigma", 0.1, 3.0, 1.0)],
    },
    "uniform": {
        "fn": _uniform,
        "name": "Uniform",
        "formula": "U(a, b)",
        "discrete": False,
        "params": [("lo", -3.0, 0.0, -1.0), ("hi", 0.0, 3.0, 1.0)],
    },
    "exponential": {
        "fn": _exponential,
        "name": "Exponential",
        "formula": "lambda * e^(-lambda x)",
        "discrete": False,
        "params": [("rate", 0.1, 5.0, 1.0)],
    },
    "laplace": {
        "fn": _laplace,
        "name": "Laplace",
        "formula": "(1/2b) e^(-|x-mu|/b)",
        "discrete": False,
        "params": [("mu", -2.0, 2.0, 0.0), ("b", 0.1, 2.0, 0.7)],
    },
    "binomial": {
        "fn": _binomial,
        "name": "Binomial",
        "formula": "C(n,k) p^k (1-p)^(n-k)",
        "discrete": True,
        "params": [("n", 1, 40, 20), ("p", 0.01, 0.99, 0.5)],
    },
    "poisson": {
        "fn": _poisson,
        "name": "Poisson",
        "formula": "lambda^k e^-lambda / k!",
        "discrete": True,
        "params": [("rate", 0.1, 20.0, 4.0)],
    },
}


def compute(params: dict[str, Any]) -> dict[str, Any]:
    dist_id = params.get("distribution", "gaussian")
    if dist_id not in DISTRIBUTIONS:
        dist_id = "gaussian"
    spec = DISTRIBUTIONS[dist_id]
    discrete: bool = spec["discrete"]

    # Collect parameter values with bounds
    pvals: dict[str, float] = {}
    for name, lo, hi, default in spec["params"]:
        v = float(params.get(name, default))
        pvals[name] = max(lo, min(hi, v))

    if discrete:
        x_max = 40
        x = np.arange(0, x_max + 1, dtype=float)
        y = spec["fn"](x, *pvals.values())
        # histogram samples
        rng = np.random.default_rng(int(params.get("seed", 0)))
        if dist_id == "binomial":
            samples = rng.binomial(int(pvals["n"]), pvals["p"], size=500)
        else:
            samples = rng.poisson(pvals["rate"], size=500)
        counts, edges = np.histogram(samples, bins=range(x_max + 2), density=True)
        domain = {
            "x": [-0.5, x_max + 0.5],
            "y": [0.0, float(max(y.max(), counts.max()) * 1.15)],
        }
        return {
            "discrete": True,
            "distribution": dist_id,
            "name": spec["name"],
            "formula": spec["formula"],
            "x": x.tolist(),
            "y": y.round(6).tolist(),
            "histEdges": edges.tolist(),
            "histCounts": counts.round(6).tolist(),
            "domain": domain,
            "stats": {
                "mean": round(float(samples.mean()), 3),
                "std": round(float(samples.std()), 3),
                "samples": int(len(samples)),
            },
        }

    # Continuous
    x = np.linspace(-6, 6, 300)
    y = spec["fn"](x, *pvals.values())
    rng = np.random.default_rng(int(params.get("seed", 0)))
    if dist_id == "gaussian":
        samples = rng.normal(pvals.get("mu", 0), pvals.get("sigma", 1), 500)
    elif dist_id == "uniform":
        samples = rng.uniform(pvals["lo"], pvals["hi"], 500)
    elif dist_id == "exponential":
        samples = rng.exponential(1.0 / pvals["rate"], 500)
    else:
        samples = rng.laplace(pvals.get("mu", 0), pvals.get("b", 0.7), 500)
    counts, edges = np.histogram(samples, bins=30, range=(-6, 6), density=True)
    return {
        "discrete": False,
        "distribution": dist_id,
        "name": spec["name"],
        "formula": spec["formula"],
        "x": x.round(6).tolist(),
        "y": y.round(6).tolist(),
        "histEdges": edges.round(4).tolist(),
        "histCounts": counts.round(6).tolist(),
        "domain": {"x": [-6.0, 6.0], "y": [0.0, float(y.max() * 1.15)]},
        "stats": {
            "mean": round(float(samples.mean()), 3),
            "std": round(float(samples.std()), 3),
            "samples": int(len(samples)),
        },
    }
