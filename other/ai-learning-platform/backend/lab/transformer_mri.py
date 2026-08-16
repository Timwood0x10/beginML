"""
Transformer MRI — scan the inside of a Transformer as an experiment
(plan §13, P1, simplified).

Metaphor: an MRI scan of the model. We sweep a slice plane through the
training dynamics — loss, representation entropy, gradient norm — across
layers (the "body") and training steps (the "time axis"). Healthy signs:
loss converging, entropy rising (information accumulating), gradients
staying bounded. A gradient-vanishing pathology shows up as near-zero norms
in deep layers.

SIMULATION MODE: the scan data is a deterministic synthetic model of
training dynamics (exponential loss decay + entropy growth + a gradient
pathology), NOT a real Transformer run.

Simulation contract: realtime + seeded.

No-LLM principle: every number is computed; health verdicts are rules over
the computed scan.
"""

from typing import Any

import numpy as np

N_LAYERS = 12
N_STEPS = 100

CHANNELS = ("loss", "entropy", "grad_norm")


def _channel_scan(channel: str, seed: int) -> np.ndarray:
    """[layer, step] intensity for one channel. Deterministic synthetic."""
    rng = np.random.default_rng(seed)
    layers = np.arange(N_LAYERS)[:, None]
    steps = np.arange(N_STEPS)[None, :]

    if channel == "loss":
        # Exponential decay; shallow layers converge faster (smaller tail).
        base = 0.55 + 0.35 * np.exp(-steps / 22.0)
        layer_term = 0.05 * (layers / N_LAYERS)
        noise = rng.normal(0, 0.02, size=(N_LAYERS, N_STEPS))
        return np.clip(base + layer_term + noise, 0.0, 1.2)

    if channel == "entropy":
        # Representation entropy rises as the model differentiates tokens.
        base = 0.6 + 0.55 * (1 - np.exp(-steps / 30.0))
        layer_term = 0.08 * np.sin(layers * 0.7)  # mid layers most expressive
        noise = rng.normal(0, 0.03, size=(N_LAYERS, N_STEPS))
        return np.clip(base + layer_term + noise, 0.0, 1.4)

    # grad_norm — early spikes settle; DEEP layers collapse (vanishing gradient).
    base = 1.6 * np.exp(-steps / 18.0) + 0.25
    depth_fade = np.exp(-0.35 * layers)  # deep layers lose gradient strength
    noise = rng.normal(0, 0.06, size=(N_LAYERS, N_STEPS))
    return np.clip(base * depth_fade + noise, 0.0, 2.5)


def _health(channel: str, scan: np.ndarray) -> dict[str, Any]:
    first_half = scan[:, : N_STEPS // 2].mean()
    second_half = scan[:, N_STEPS // 2 :].mean()
    deep = scan[-3:].mean()
    shallow = scan[:3].mean()

    if channel == "loss":
        return {
            "label": "loss_converged",
            "ok": bool(second_half < first_half),
            "detail": f"loss {first_half:.3f} → {second_half:.3f}",
        }
    if channel == "entropy":
        return {
            "label": "entropy_rising",
            "ok": bool(second_half > first_half),
            "detail": f"entropy {first_half:.3f} → {second_half:.3f}",
        }
    return {
        "label": "grad_stable",
        "ok": bool(deep > 0.05),  # deep layers still receiving gradient
        "detail": f"deep {deep:.3f} vs shallow {shallow:.3f}",
    }


def compute(params: dict[str, Any]) -> dict[str, Any]:
    channel = str(params.get("scan", "loss"))
    if channel not in CHANNELS:
        channel = "loss"
    current_layer = int(min(max(int(params.get("layer", 6)), 0), N_LAYERS - 1))
    current_step = int(min(max(int(params.get("step", N_STEPS - 1)), 0), N_STEPS - 1))
    seed = int(params.get("seed", 3))
    repair = float(params.get("repair", 1.0))  # residual scaling (fix) factor

    scan = _channel_scan(channel, seed)  # [layer, step]

    # "Fix the pathology" mechanic: on the grad_norm channel, amplify the
    # deep layers (residual scaling, like a gradient shortcut). Deterministic
    # and computed — the health verdict is recomputed AFTER the fix so the
    # learner can see the pathology heal.
    applied = False
    if channel == "grad_norm" and repair > 1.0:
        scan = scan.copy()
        deep = max(1, int(N_LAYERS * 0.66))  # deep third (L >= 8)
        scan[deep:] = scan[deep:] * repair
        applied = True

    return {
        "scan": channel,
        "steps": list(range(N_STEPS)),
        "layers": list(range(N_LAYERS)),
        "heatmap": np.round(scan, 4).tolist(),
        "current_layer": current_layer,
        "current_step": current_step,
        "layer_curve": np.round(scan[current_layer], 4).tolist(),
        "step_profile": np.round(scan[:, current_step], 4).tolist(),
        "health": _health(channel, scan),
        "channels": list(CHANNELS),
        "repair": round(repair, 2),
        "repair_applied": applied,
        "seed": seed,
        "simulation_mode": True,
        "provenance": f"seeded(synthetic MRI scan, seed {seed})",
    }
