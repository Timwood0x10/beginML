"""
1D convolution visualization — compute each sliding-window product and the
resulting output. The frontend animates the kernel across the input.
"""

from typing import Any

import numpy as np

KERNELS: dict[str, list[float]] = {
    "identity": [0, 1, 0],
    "edge-detect": [-1, 0, 1],
    "sharpen": [0, -1, 0, -1, 5, -1, 0, -1, 0],
    "box-blur": [1 / 9] * 9,
    "gaussian": [0.06, 0.12, 0.20, 0.24, 0.20, 0.12, 0.06],
    "emboss": [-2, -1, 0, -1, 1, 1, 0, 1, 2],
}


def _default_input(n: int) -> list[float]:
    """A noisy step signal so the kernels produce visible effects."""
    rng = np.random.default_rng(42)
    x = np.linspace(0, 4 * np.pi, n)
    signal = np.where(x > 2 * np.pi, 1.0, 0.0)
    signal += 0.3 * np.sin(2 * x) + 0.05 * rng.normal(size=n)
    return np.round(signal, 4).tolist()


def compute(params: dict[str, Any]) -> dict[str, Any]:
    kernel_id = params.get("kernel", "edge-detect")
    if kernel_id not in KERNELS:
        kernel_id = "edge-detect"
    kernel = np.array(KERNELS[kernel_id], dtype=float)

    n = int(params.get("length", 32))
    raw_input = params.get("input")
    if isinstance(raw_input, list) and len(raw_input) >= len(kernel):
        signal = np.array(raw_input, dtype=float)
    else:
        signal = np.array(_default_input(n), dtype=float)

    k = len(kernel)
    out_len = len(signal) - k + 1
    output = np.zeros(out_len)
    windows: list[dict[str, Any]] = []

    for i in range(out_len):
        window = signal[i : i + k]
        products = window * kernel
        value = float(np.sum(products))
        output[i] = value
        windows.append(
            {
                "position": i,
                "window": np.round(window, 4).tolist(),
                "products": np.round(products, 4).tolist(),
                "value": round(value, 4),
            }
        )

    return {
        "input": np.round(signal, 4).tolist(),
        "kernel": np.round(kernel, 4).tolist(),
        "output": np.round(output, 4).tolist(),
        "windows": windows,
        "kernelName": kernel_id,
        "kernelSize": k,
        "available": [
            {"id": kid, "name": kid.replace("-", " ").title()} for kid in KERNELS
        ],
        "domain": {
            "x": [0, max(len(signal), out_len) - 1],
            "y": [
                float(min(signal.min(), output.min()) - 0.2),
                float(max(signal.max(), output.max()) + 0.2),
            ],
        },
    }
