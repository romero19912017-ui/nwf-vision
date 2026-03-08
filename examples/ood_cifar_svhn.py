# -*- coding: utf-8 -*-
"""OOD detection: CIFAR-10 (in) vs SVHN (out) using NWF potential.

Train ConvVAE on CIFAR-10, build charges. Higher potential = in-distribution.
Visualizes ROC, histograms, example images.
Run: python ood_cifar_svhn.py [--epochs 3] [--save results/ood_cifar_svhn.png]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from torchvision.datasets import CIFAR10, SVHN
from torchvision.transforms import ToTensor

from nwf import potential_batch
from nwf.vision import ConvVAEEncoder

if "--save" in sys.argv or os.environ.get("MPLBACKEND"):
    import matplotlib
    matplotlib.use("Agg")


def main() -> None:
    parser = argparse.ArgumentParser(description="OOD: CIFAR-10 vs SVHN via potential")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-in-train", type=int, default=5000)
    parser.add_argument("--n-in-test", type=int, default=500)
    parser.add_argument("--n-ood", type=int, default=1000)
    parser.add_argument("--save", type=str, default="")
    args = parser.parse_args()

    print("Loading CIFAR-10 (in) and SVHN (out)...")
    ds_cifar = CIFAR10(root="./data", train=True, download=True, transform=ToTensor())
    X_in_train = np.stack([ds_cifar[i][0].numpy() for i in range(min(args.n_in_train, len(ds_cifar)))])
    ds_cifar_test = CIFAR10(root="./data", train=False, download=True, transform=ToTensor())
    X_in_test = np.stack([ds_cifar_test[i][0].numpy() for i in range(min(args.n_in_test, len(ds_cifar_test)))])
    ds_svhn = SVHN(root="./data", split="test", download=True, transform=ToTensor())
    X_ood = np.stack([ds_svhn[i][0].numpy() for i in range(min(args.n_ood, len(ds_svhn)))])
    print(f"CIFAR train: {len(X_in_train)}, test: {len(X_in_test)}, SVHN: {len(X_ood)}")

    print("Training ConvVAE on CIFAR-10...")
    enc = ConvVAEEncoder(input_shape=(3, 32, 32), latent_dim=args.latent_dim)
    enc.fit(X_in_train, epochs=args.epochs, batch_size=args.batch_size)

    z_in, s_in = enc.encode(X_in_train)
    if z_in.ndim == 1:
        z_in = z_in.reshape(1, -1)
        s_in = s_in.reshape(1, -1)
    z_all = z_in.astype(np.float64)
    s_all = np.maximum(s_in, 1e-6).astype(np.float64)

    print("Encoding test sets...")
    z_cifar = enc.encode(X_in_test)[0]
    z_svhn = enc.encode(X_ood)[0]
    if z_cifar.ndim == 1:
        z_cifar = z_cifar.reshape(1, -1)
    if z_svhn.ndim == 1:
        z_svhn = z_svhn.reshape(1, -1)

    phi_in = potential_batch(z_cifar, z_all, s_all)
    phi_out = potential_batch(z_svhn, z_all, s_all)

    y_true = np.concatenate([np.ones(len(phi_in)), np.zeros(len(phi_out))])
    y_score = np.concatenate([phi_in, phi_out])
    auroc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    print(f"OOD AUROC (CIFAR vs SVHN): {auroc:.3f}")

    if args.save:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(14, 5))
        gs = GridSpec(2, 8, figure=fig)
        ax1 = fig.add_subplot(gs[0, :4])
        ax2 = fig.add_subplot(gs[0, 4:])
        ax1.hist(phi_in, bins=30, alpha=0.7, label="CIFAR-10 (in)", color="C0")
        ax1.hist(phi_out, bins=30, alpha=0.7, label="SVHN (OOD)", color="C1")
        ax1.set_xlabel("Potential")
        ax1.set_ylabel("Count")
        ax1.set_title("Potential distribution")
        ax1.legend()
        ax2.plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
        ax2.plot([0, 1], [0, 1], "k--")
        ax2.set_xlabel("FPR")
        ax2.set_ylabel("TPR")
        ax2.set_title("ROC curve")
        ax2.legend()
        idx_high = np.argsort(phi_in)[-4:]
        idx_low = np.argsort(phi_out)[:4]
        for j, idx in enumerate(idx_high):
            ax = fig.add_subplot(gs[1, j])
            im = X_in_test[idx].transpose(1, 2, 0)
            ax.imshow(np.clip(im, 0, 1))
            ax.set_title(f"in phi={phi_in[idx]:.1f}")
            ax.axis("off")
        for j, idx in enumerate(idx_low):
            ax = fig.add_subplot(gs[1, j + 4])
            im = X_ood[idx].transpose(1, 2, 0)
            ax.imshow(np.clip(im, 0, 1))
            ax.set_title(f"OOD phi={phi_out[idx]:.1f}")
            ax.axis("off")
        plt.tight_layout()
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {args.save}")
    print("Done.")


if __name__ == "__main__":
    main()
