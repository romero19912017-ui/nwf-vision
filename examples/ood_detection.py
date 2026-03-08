# -*- coding: utf-8 -*-
"""OOD detection: CIFAR-10 (in) vs SVHN (out) using NWF potential.

Train ConvVAE on CIFAR-10, build charges. Lower potential = OOD.
Run: python ood_detection.py
"""
from __future__ import annotations

import numpy as np
from torchvision.datasets import CIFAR10, SVHN
from torchvision.transforms import ToTensor

from nwf import Charge, Field, potential, potential_batch
from nwf.vision import ConvVAEEncoder


def get_cifar(root: str = "./data") -> tuple:
    ds = CIFAR10(root=root, train=True, download=True, transform=ToTensor())
    X = np.stack([ds[i][0].numpy() for i in range(min(5000, len(ds)))])
    return X


def get_svhn(root: str = "./data") -> tuple:
    ds = SVHN(root=root, split="test", download=True, transform=ToTensor())
    X = np.stack([ds[i][0].numpy() for i in range(min(1000, len(ds)))])
    return X


def main() -> None:
    print("Loading CIFAR-10 (in) and SVHN (out)...")
    X_in = get_cifar()
    X_out = get_svhn()
    print(f"CIFAR-10: {len(X_in)}, SVHN: {len(X_out)}")

    print("Training ConvVAE on CIFAR-10...")
    enc = ConvVAEEncoder(input_shape=(3, 32, 32), latent_dim=32)
    enc.fit(X_in, epochs=3, batch_size=128)

    print("Building in-distribution charges (class means)...")
    z_in, s_in = enc.encode(X_in)
    if z_in.ndim == 1:
        z_in = z_in.reshape(1, -1)
        s_in = s_in.reshape(1, -1)
    charges = [
        Charge(z=z_in[i].astype(np.float64), sigma=np.maximum(s_in[i], 1e-6).astype(np.float64))
        for i in range(len(z_in))
    ][:500]
    z_all = np.stack([c.z for c in charges])
    s_all = np.stack([c.sigma for c in charges])

    print("Encoding test sets...")
    z_cifar, s_cifar = enc.encode(X_in[5000:5500] if len(X_in) > 5500 else X_in[-500:])
    z_svhn, s_svhn = enc.encode(X_out)
    if z_cifar.ndim == 1:
        z_cifar = z_cifar.reshape(1, -1)
        s_cifar = s_cifar.reshape(1, -1)
    if z_svhn.ndim == 1:
        z_svhn = z_svhn.reshape(1, -1)
        s_svhn = s_svhn.reshape(1, -1)

    print("Computing potential (higher = in-distribution)...")
    phi_in = potential_batch(z_cifar, z_all, s_all)
    phi_out = potential_batch(z_svhn, z_all, s_all)

    from sklearn.metrics import roc_auc_score

    y_true = np.concatenate([np.ones(len(phi_in)), np.zeros(len(phi_out))])
    y_score = np.concatenate([phi_in, phi_out])
    auroc = roc_auc_score(y_true, y_score)
    print(f"OOD AUROC (CIFAR vs SVHN): {auroc:.3f}")

    threshold = np.median(y_score)
    pred = (y_score >= threshold).astype(int)
    acc = (pred == y_true).mean()
    print(f"Accuracy at median threshold: {acc:.3f}")
    print("Done.")


if __name__ == "__main__":
    main()
