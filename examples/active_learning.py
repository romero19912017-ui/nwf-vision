# -*- coding: utf-8 -*-
"""Active learning on CIFAR-10: uncertainty sampling vs random.

Select samples by trace(sigma), label them, add to Field. Compare with random.
Run: python active_learning.py
"""
from __future__ import annotations

import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from nwf import Charge, Field
from nwf.vision import ConvVAEEncoder


def get_cifar(root: str = "./data") -> tuple:
    ds = CIFAR10(root=root, train=True, download=True, transform=ToTensor())
    X = np.stack([ds[i][0].numpy() for i in range(len(ds))])
    y = np.array([ds[i][1] for i in range(len(ds))])
    return X, y


def run_active(
    enc: ConvVAEEncoder,
    X: np.ndarray,
    y: np.ndarray,
    n_labeled: int,
    strategy: str = "uncertainty",
    seed: int = 42,
) -> float:
    """Label n_labeled samples by strategy, train Field, evaluate on rest."""
    rng = np.random.RandomState(seed)
    n = len(X)
    if strategy == "random":
        idx_labeled = rng.choice(n, size=n_labeled, replace=False)
    else:
        z_all, s_all = enc.encode(X)
        if z_all.ndim == 1:
            z_all = z_all.reshape(1, -1)
            s_all = s_all.reshape(1, -1)
        uncertainty = np.sum(s_all, axis=1)
        idx_labeled = np.argsort(-uncertainty)[:n_labeled]

    field = Field()
    for i in idx_labeled:
        z, s = enc.encode(X[i : i + 1])
        z = z.ravel()
        s = np.maximum(s.ravel(), 1e-6)
        field.add(Charge(z=z, sigma=s), labels=[int(y[i])], ids=[i])

    idx_test = np.array([j for j in range(n) if j not in idx_labeled])
    correct = 0
    for i in idx_test[:500]:
        z, s = enc.encode(X[i : i + 1])
        z = z.ravel()
        s = np.maximum(s.ravel(), 1e-6)
        q = Charge(z=z, sigma=s)
        _, _, labs = field.search(q, k=5)
        votes = np.bincount(np.array(labs[0]), minlength=10)
        pred = np.argmax(votes)
        if pred == y[i]:
            correct += 1
    return correct / min(500, len(idx_test))


def main() -> None:
    print("Loading CIFAR-10...")
    X, y = get_cifar()
    print(f"Total samples: {len(X)}")

    print("Training ConvVAE (3 epochs)...")
    enc = ConvVAEEncoder(input_shape=(3, 32, 32), latent_dim=32)
    enc.fit(X, epochs=3, batch_size=128)

    n_labeled = 200
    print(f"Active learning: {n_labeled} labeled samples")
    acc_unc = run_active(enc, X, y, n_labeled, strategy="uncertainty")
    acc_rand = run_active(enc, X, y, n_labeled, strategy="random")
    print(f"Uncertainty sampling accuracy: {acc_unc:.3f}")
    print(f"Random sampling accuracy: {acc_rand:.3f}")
    print("Done.")


if __name__ == "__main__":
    main()
