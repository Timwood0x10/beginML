"""
Transformer training simulation for the interactive lab.

A tiny decoder-only Transformer (learnable token + positional embeddings,
multi-head self-attention with optional causal masking, a two-layer FFN and
a final LM head) is trained from scratch on a "predict the previous token"
task.  The task forces the model to learn a clean one-step-back attention
pattern, so the per-head attention heatmaps become interpretable after
training and the loss curve shows real convergence.

All math is plain NumPy with hand-derived gradients, so the lab needs no
deep-learning framework.  Project rules: every function <= 120 lines, the
file <= 1000 lines, all comments in English.
"""

from typing import Any

import numpy as np

D_MODEL = 16  # embedding / model width. Each head gets dk = D_MODEL // heads,
# so heads MUST divide D_MODEL (valid choices: 1, 2, 4, 8, 16).
VOCAB = 12  # distinct tokens in the toy vocabulary


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along one axis."""
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    """Layer norm over the last axis; returns (out, cache)."""
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xn = (x - mu) * inv
    return xn * gamma + beta, (x, mu, inv, gamma)


def _layernorm_backward(dout: np.ndarray, cache):
    """Backward through layer norm; returns (dx, dgamma, dbeta).

    gamma/beta are per-feature vectors broadcast over positions, so their
    gradients sum over the position axis (axis 0); dx is normalized over the
    feature axis (last axis).
    """
    x, mu, inv, gamma = cache
    xn = (x - mu) * inv
    dgamma = np.sum(dout * xn, axis=0)
    dbeta = np.sum(dout, axis=0)
    dxn = dout * gamma
    m = np.mean(dxn, axis=-1, keepdims=True)
    cov = np.mean(dxn * xn, axis=-1, keepdims=True)
    dx = (dxn - m - xn * cov) * inv
    return dx, dgamma, dbeta


def _gelu(x: np.ndarray) -> np.ndarray:
    """Tanh-approximated GELU used as the FFN activation."""
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))


def _gelu_deriv(x: np.ndarray) -> np.ndarray:
    """Derivative of the tanh-approximated GELU."""
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x**3)
    t = np.tanh(inner)
    d_inner = c * (1.0 + 3 * 0.044715 * x**2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * d_inner


def _cross_entropy(logits: np.ndarray, targets: np.ndarray, ignore: np.ndarray):
    """Mean cross-entropy; returns (loss, dlogits) with `ignore` positions zeroed."""
    p = _stable_softmax(logits, axis=-1)
    probs = p[np.arange(logits.shape[0]), targets]
    loss = -np.mean(np.log(probs[ignore] + 1e-9)) if ignore.any() else 0.0
    dlogits = p.copy()
    dlogits[np.arange(logits.shape[0]), targets] -= 1.0
    dlogits[~ignore] = 0.0
    n_keep = max(1, int(ignore.sum()))
    return float(loss), dlogits / n_keep


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class NanoTransformer:
    """Minimal decoder-only transformer with hand-written forward/backward."""

    def __init__(self, heads: int, layers: int, max_len: int, seed: int = 0):
        d = D_MODEL
        self.d = d
        self.h = heads
        self.dk = d // heads
        self.l = layers
        rng = np.random.default_rng(seed)
        scale = 0.1
        self.Wte = rng.normal(0, scale, (VOCAB, d))  # token embeddings
        self.Wpe = rng.normal(0, scale, (max_len, d))  # positional encodings
        self.Whead = rng.normal(0, scale, (d, VOCAB))  # unshared LM head
        self.blocks: list[dict[str, np.ndarray]] = []
        for _ in range(layers):
            block = {
                "Wq": rng.normal(0, 1 / np.sqrt(d), (d, d)),
                "Wk": rng.normal(0, 1 / np.sqrt(d), (d, d)),
                "Wv": rng.normal(0, 1 / np.sqrt(d), (d, d)),
                "Wo": rng.normal(0, 1 / np.sqrt(d), (d, d)),
                "W1": rng.normal(0, 1 / np.sqrt(d), (d, 2 * d)),
                "W2": rng.normal(0, 1 / np.sqrt(d), (2 * d, d)),
                "ln1g": np.ones(d),
                "ln1b": np.zeros(d),
                "ln2g": np.ones(d),
                "ln2b": np.zeros(d),
            }
            self.blocks.append(block)

    # -- forward ---------------------------------------------------------

    def forward(self, x: np.ndarray, causal: bool = True):
        """Forward pass; returns (logits, cache) where cache drives backward."""
        n = x.shape[0]
        h = self.Wte[x] + self.Wpe[:n]
        cache: dict[str, Any] = {
            "x": x,
            "h0": h,
            "attn": [],
            "ln": [],
            "ffn": [],
            "qk": [],
        }
        mask = np.triu(np.full((n, n), -1e9), k=1) if causal else None

        for blk in self.blocks:
            # attention sub-block
            h1n, c1 = _layernorm(h, blk["ln1g"], blk["ln1b"])
            q = h1n @ blk["Wq"]
            k = h1n @ blk["Wk"]
            v = h1n @ blk["Wv"]
            qh = q.reshape(n, self.h, self.dk).transpose(1, 0, 2)
            kh = k.reshape(n, self.h, self.dk).transpose(1, 0, 2)
            vh = v.reshape(n, self.h, self.dk).transpose(1, 0, 2)
            scores = qh @ kh.transpose(0, 2, 1) / np.sqrt(self.dk)
            if mask is not None:
                scores = scores + mask
            attn = _stable_softmax(scores, axis=-1)  # (H, n, n)
            ctx = attn @ vh  # (H, n, dk)
            ctx_cat = ctx.transpose(1, 0, 2).reshape(n, self.d)
            attn_out = ctx_cat @ blk["Wo"]
            h1 = h + attn_out

            # feed-forward sub-block
            h2n, c2 = _layernorm(h1, blk["ln2g"], blk["ln2b"])
            f1 = h2n @ blk["W1"]
            act = _gelu(f1)
            f2 = act @ blk["W2"]
            h = h1 + f2

            cache["attn"].append(attn)
            cache["ln"].append((c1, c2, h1n))
            cache["ffn"].append((h2n, f1, act, vh))
            cache["qk"].append((qh, kh))

        logits = h @ self.Whead
        cache["h"] = h
        return logits, cache

    # -- backward --------------------------------------------------------

    def backward(self, dlogits: np.ndarray, cache: dict) -> dict[str, np.ndarray]:
        """Backprop over all parameters; returns name -> gradient."""
        grads: dict[str, np.ndarray] = {}
        dh = dlogits @ self.Whead.T
        grads["Whead"] = cache["h"].T @ dlogits

        for i in range(self.l - 1, -1, -1):
            blk = self.blocks[i]
            h2n, f1, act, vh = cache["ffn"][i]
            c1, c2, h1n = cache["ln"][i]
            n = dh.shape[0]

            # FFN: h_out = h1 + f2, so h1 receives the FFN gradient AND `dh`
            # directly through the residual connection.
            df2 = dh
            grads[f"b{i}_W2"] = act.T @ df2
            dact = df2 @ blk["W2"].T
            df1 = dact * _gelu_deriv(f1)
            grads[f"b{i}_W1"] = h2n.T @ df1
            dh2n = df1 @ blk["W1"].T
            dh1_from_ln2, dg2, db2 = _layernorm_backward(dh2n, c2)
            grads[f"b{i}_ln2g"] = dg2
            grads[f"b{i}_ln2b"] = db2
            dh1 = dh1_from_ln2 + dh

            # attention sub-block: h1 = h_in + attn_out
            dattn_out = dh1
            dctx_cat = dattn_out @ blk["Wo"].T
            ctx_cat = cache["attn"][i] @ vh
            ctx_cat = ctx_cat.transpose(1, 0, 2).reshape(n, self.d)
            grads[f"b{i}_Wo"] = ctx_cat.T @ dattn_out

            dctx = dctx_cat.reshape(n, self.h, self.dk).transpose(1, 0, 2)
            attn = cache["attn"][i]
            dattn = dctx @ vh.transpose(0, 2, 1)
            dvh = attn.transpose(0, 2, 1) @ dctx
            dscores = attn * (dattn - np.sum(dattn * attn, axis=-1, keepdims=True))
            qh, kh = cache["qk"][i]
            dqh = dscores @ kh / np.sqrt(self.dk)
            dkh = dscores.transpose(0, 2, 1) @ qh / np.sqrt(self.dk)
            dv = dvh.transpose(1, 0, 2).reshape(n, self.d)
            dq = dqh.transpose(1, 0, 2).reshape(n, self.d)
            dk = dkh.transpose(1, 0, 2).reshape(n, self.d)
            grads[f"b{i}_Wq"] = h1n.T @ dq
            grads[f"b{i}_Wk"] = h1n.T @ dk
            grads[f"b{i}_Wv"] = h1n.T @ dv
            dh1n = dq @ blk["Wq"].T + dk @ blk["Wk"].T + dv @ blk["Wv"].T
            dh_h, dg1, db1 = _layernorm_backward(dh1n, c1)
            grads[f"b{i}_ln1g"] = dg1
            grads[f"b{i}_ln1b"] = db1
            dh = dh_h + dattn_out  # residual: h1 = h_in + attn_out

        # embeddings
        x = cache["x"]
        dWte = np.zeros_like(self.Wte)
        dWpe = np.zeros_like(self.Wpe)
        np.add.at(dWte, x, dh)
        dWpe[: dh.shape[0]] += dh
        grads["Wte"] = dWte
        grads["Wpe"] = dWpe
        return grads

    # -- update ----------------------------------------------------------

    def step(self, grads: dict[str, np.ndarray], lr: float) -> None:
        """SGD update with a tiny L2 penalty (weight decay) for stability."""
        for name, g in grads.items():
            param = self._param(name)
            param -= lr * (g + 1e-4 * param)

    def _param(self, name: str) -> np.ndarray:
        """Resolve a flat parameter name back to its array."""
        if name == "Wte":
            return self.Wte
        if name == "Wpe":
            return self.Wpe
        if name == "Whead":
            return self.Whead
        i = int(name[1])
        return self.blocks[i][name[3:]]


# ---------------------------------------------------------------------------
# Task + driver
# ---------------------------------------------------------------------------


def _make_batch(
    rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One training example: predict the previous token (position 0 ignored)."""
    x = rng.integers(0, VOCAB, size=n)
    y = np.roll(x, 1)  # y[t] = x[t-1]
    y[0] = 0
    ignore = np.ones(n, dtype=bool)
    ignore[0] = False
    return x, y, ignore


def _train(
    model: NanoTransformer,
    rng: np.random.Generator,
    epochs: int,
    lr: float,
    causal: bool,
    n: int,
) -> list[float]:
    """Run SGD for `epochs` steps; returns the loss recorded per epoch."""
    losses: list[float] = []
    for _ in range(epochs):
        x, y, ignore = _make_batch(rng, n)
        logits, cache = model.forward(x, causal)
        loss, dlogits = _cross_entropy(logits, y, ignore)
        grads = model.backward(dlogits, cache)
        model.step(grads, lr)
        losses.append(round(loss, 4))
    return losses


def compute(params: dict[str, Any]) -> dict[str, Any]:
    """Lab entry point: train a tiny transformer and report loss + attention."""
    n = int(params.get("tokens", 8))
    heads = int(params.get("heads", 2))
    layers = int(params.get("layers", 2))
    lr = float(params.get("lr", 0.05))
    epochs = int(params.get("epochs", 100))
    causal = bool(params.get("causal", True))
    seed = int(params.get("seed", 3))
    # heads must divide D_MODEL so dk = D_MODEL // heads is an integer; snap
    # any invalid value (e.g. 3) down to the nearest valid divisor (2).
    heads = max(1, min(heads, D_MODEL))
    while D_MODEL % heads != 0:
        heads -= 1

    rng = np.random.default_rng(seed)
    model = NanoTransformer(heads=heads, layers=layers, max_len=n + 2, seed=seed)
    losses = _train(model, rng, epochs=epochs, lr=lr, causal=causal, n=n)

    # Final attention pattern per layer and head (H, n, n).
    x, _y, _ignore = _make_batch(rng, n)
    _, cache = model.forward(x, causal)
    attn = [np.round(a, 4).tolist() for a in cache["attn"]]

    return {
        "tokens": [f"t{i}" for i in range(n)],
        "n": n,
        "heads": heads,
        "layers": layers,
        "lr": lr,
        "epochs": epochs,
        "causal": causal,
        "losses": losses,
        "finalLoss": losses[-1] if losses else 0.0,
        "attn": attn,  # layers -> heads -> n x n
        "task": "predict the previous token (next-token prediction)",
    }
