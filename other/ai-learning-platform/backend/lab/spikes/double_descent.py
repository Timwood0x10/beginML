"""
Spike: does random-feature ridge regression produce a clean double descent?

Plan §10.3 freezes this as a REQUIRED spike before any frontend work: the
random-Fourier-features recipe may or may not give the textbook curve — the
shape depends on data generation, feature distribution, regularization, the
interpolation regime, sample/feature ratio, noise and solver.

This script sweeps a few knobs, prints the resulting train/test error curves
and asserts (visually + numerically) that we get:
  * low capacity  -> both errors high (underfit),
  * capacity ~ n  -> interpolation threshold with a test-error peak,
  * capacity > n  -> test error drops again (overparameterized).

Usage:  .venv/bin/python lab/spikes/double_descent.py
"""

import numpy as np


def run_sweep(
    n: int = 64,
    d: int = 20,
    p_max: int = 192,
    lam: float = 1e-4,
    noise: float = 0.2,
    seed: int = 0,
    fourier: bool = True,
    n_test: int = 4000,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    # --- data: linear teacher + label noise ------------------------------
    w_true = rng.normal(0, 1.0, size=d)
    X_tr = rng.normal(0, 1.0, size=(n, d))
    y_tr = X_tr @ w_true + noise * rng.normal(0, 1.0, size=n)
    X_te = rng.normal(0, 1.0, size=(n_test, d))
    y_te = X_te @ w_true + noise * rng.normal(0, 1.0, size=n_test)
    # reference test noise floor
    var_noise = noise * noise

    # --- random features ------------------------------------------------
    if fourier:
        # random Fourier features: z = sqrt(2/p) * cos(X @ W + b), W ~ N(0, gamma^2)
        gamma = 1.0
    else:
        gamma = 1.0

    caps: list[int] = []
    train_err: list[float] = []
    test_err: list[float] = []
    peak: float = -1.0
    peak_cap: int = -1

    for p in range(4, p_max + 1, 4):
        if fourier:
            W = rng.normal(0, gamma, size=(d, p))
            b = rng.uniform(0, 2 * np.pi, size=p)
            phi = lambda X: np.sqrt(2.0 / p) * np.cos(X @ W + b)
            Phi_tr = phi(X_tr)
            Phi_te = phi(X_te)
        else:
            A = rng.normal(0, 1.0 / np.sqrt(p), size=(p, d))
            Phi_tr = X_tr @ A.T
            Phi_te = X_te @ A.T

        # ridge solve
        G = Phi_tr.T @ Phi_tr + lam * np.eye(p)
        theta = np.linalg.solve(G, Phi_tr.T @ y_tr)

        tr_e = float(np.mean((Phi_tr @ theta - y_tr) ** 2))
        te_e = float(np.mean((Phi_te @ theta - y_te) ** 2) - var_noise)  # excess risk
        caps.append(p)
        train_err.append(tr_e)
        test_err.append(max(te_e, 0.0))

        if te_e > peak:
            peak = te_e
            peak_cap = p

    return caps, np.array(train_err), np.array(test_err), peak, peak_cap


def main() -> None:
    print("=== double-descent spike: random features + ridge ===\n")
    best: tuple[float, int, dict] | None = None

    for n in (48, 64, 96):
        for noise in (0.1, 0.2, 0.4):
            for lam in (1e-5, 1e-4, 1e-3):
                for fourier in (True, False):
                    caps, tr, te, peak, peak_cap = run_sweep(
                        n=n, noise=noise, lam=lam, fourier=fourier, seed=3
                    )
                    # success criteria:
                    # 1) a distinct peak exists near/interpolation: peak_cap >= 0.6*n
                    # 2) peak is meaningfully higher than both sides
                    i_peak = int(np.argmax(te))
                    left = float(np.mean(te[max(0, i_peak - 4):i_peak]))
                    right = float(np.mean(te[i_peak + 1:i_peak + 5]))
                    ok = (te[i_peak] > 1.5 * left) and (te[i_peak] > 1.5 * right) and (caps[i_peak] >= 0.5 * n)
                    if ok:
                        ratio = te[i_peak] / max(float(te[-1]), 1e-9)
                        score = ratio
                        cfg = dict(n=n, noise=noise, lam=lam, fourier=fourier)
                        print(
                            f"n={n:3d} noise={noise:.1f} lam={lam:.0e} "
                            f"{'fourier' if fourier else 'gauss ':8s} "
                            f"peak@cap={caps[i_peak]:3d} peak={te[i_peak]:.4f} "
                            f"tail={te[-1]:.4f} ratio={ratio:6.1f}"
                        )
                        if best is None or ratio > best[0]:
                            best = (ratio, peak_cap, cfg)

    print("\n=== candidate params ===")
    if best is None:
        print("NO config produced a clean double descent — need a different recipe.")
        return
    ratio, peak_cap, cfg = best
    print(cfg, "peak-ratio", round(ratio, 1))

    # print the winning curve
    caps, tr, te, _, _ = run_sweep(**cfg, seed=7)
    print(f"\nwinning curve (seed=7): n={cfg['n']} noise={cfg['noise']} lam={cfg['lam']} "
          f"{'fourier' if cfg['fourier'] else 'gauss'}")
    for c, e_tr, e_te in zip(caps, tr, te):
        marker = " <== peak" if c == peak_cap else ""
        print(f"  cap={c:3d}  train={e_tr:.4f}  test={e_te:.4f}{marker}")


if __name__ == "__main__":
    main()
