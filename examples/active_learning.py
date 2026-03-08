# -*- coding: utf-8 -*-
"""Active learning on CIFAR-10: uncertainty sampling vs random.

Uncertainty = trace(sigma). Iteratively select most uncertain unlabeled samples,
add to labeled pool, evaluate accuracy. Compare with random strategy.
Run: python active_learning.py [--n-initial 500] [--n-per-step 100] [--save results/active.png]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from nwf import Charge, Field
from nwf.vision import ConvVAEEncoder

if "--save" in sys.argv or os.environ.get("MPLBACKEND"):
    import matplotlib
    matplotlib.use("Agg")


def get_cifar(root: str = "./data", max_samples: int | None = None) -> tuple:
    ds = CIFAR10(root=root, train=True, download=True, transform=ToTensor())
    n = min(max_samples or len(ds), len(ds))
    X = np.stack([ds[i][0].numpy() for i in range(n)])
    y = np.array([ds[i][1] for i in range(n)])
    return X, y


def evaluate(
    enc: ConvVAEEncoder,
    field: Field,
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    max_test: int = 500,
) -> float:
    correct = 0
    n = min(max_test, len(X))
    for i in range(n):
        z, s = enc.encode(X[i : i + 1])
        z = z.ravel()
        s = np.maximum(s.ravel(), 1e-6)
        q = Charge(z=z, sigma=s)
        _, _, labs = field.search(q, k=k)
        votes = np.bincount(np.array(labs[0]).astype(int), minlength=10)
        pred = int(np.argmax(votes))
        if pred == y[i]:
            correct += 1
    return correct / n if n > 0 else 0.0


def run_strategy(
    enc: ConvVAEEncoder,
    X: np.ndarray,
    y: np.ndarray,
    n_initial: int,
    n_per_step: int,
    n_steps: int,
    strategy: str,
    seed: int,
    k: int,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> list[float]:
    rng = np.random.RandomState(seed)
    n = len(X)
    idx_labeled = set(rng.choice(n, size=n_initial, replace=False).tolist())
    accuracies = []

    for step in range(n_steps):
        field = Field()
        for i in idx_labeled:
            z, s = enc.encode(X[i : i + 1])
            z = z.ravel()
            s = np.maximum(s.ravel(), 1e-6)
            field.add(Charge(z=z, sigma=s), labels=[int(y[i])], ids=[i])

        acc = evaluate(enc, field, X_test, y_test, k=k)
        accuracies.append(acc)

        if step == n_steps - 1:
            break
        idx_unlabeled = [j for j in range(n) if j not in idx_labeled]
        if len(idx_unlabeled) < n_per_step:
            break

        if strategy == "random":
            chosen = rng.choice(idx_unlabeled, size=n_per_step, replace=False)
        else:
            z_all, s_all = enc.encode(X[idx_unlabeled])
            if z_all.ndim == 1:
                z_all = z_all.reshape(1, -1)
                s_all = s_all.reshape(1, -1)
            uncertainty = np.sum(s_all, axis=1)
            top_idx = np.argsort(-uncertainty)[:n_per_step]
            chosen = np.array(idx_unlabeled)[top_idx]
        for j in chosen:
            idx_labeled.add(int(j))

    return accuracies


def main() -> None:
    parser = argparse.ArgumentParser(description="Active learning: uncertainty vs random")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-initial", type=int, default=500)
    parser.add_argument("--n-per-step", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--max-train", type=int, default=5000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="")
    args = parser.parse_args()

    print("Loading CIFAR-10...")
    X, y = get_cifar(max_samples=args.max_train)
    ds_test = CIFAR10(root="./data", train=False, download=True, transform=ToTensor())
    X_test = np.stack([ds_test[i][0].numpy() for i in range(min(1000, len(ds_test)))])
    y_test = np.array([ds_test[i][1] for i in range(min(1000, len(ds_test)))])
    print(f"Train: {len(X)}, Test: {len(X_test)}")

    print("Training ConvVAE...")
    enc = ConvVAEEncoder(input_shape=(3, 32, 32), latent_dim=args.latent_dim)
    enc.fit(X, epochs=args.epochs, batch_size=args.batch_size)

    print("Running uncertainty sampling...")
    acc_unc = run_strategy(
        enc, X, y,
        n_initial=args.n_initial,
        n_per_step=args.n_per_step,
        n_steps=args.n_steps,
        strategy="uncertainty",
        seed=args.seed,
        k=args.k,
        X_test=X_test,
        y_test=y_test,
    )
    print("Running random sampling...")
    acc_rand = run_strategy(
        enc, X, y,
        n_initial=args.n_initial,
        n_per_step=args.n_per_step,
        n_steps=args.n_steps,
        strategy="random",
        seed=args.seed,
        k=args.k,
        X_test=X_test,
        y_test=y_test,
    )

    n_labeled_points = [
        args.n_initial + step * args.n_per_step
        for step in range(len(acc_unc))
    ]
    print(f"Uncertainty final acc: {acc_unc[-1]:.4f}")
    print(f"Random final acc: {acc_rand[-1]:.4f}")

    if args.save:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(n_labeled_points, acc_unc, "o-", label="Uncertainty")
        ax.plot(n_labeled_points, acc_rand, "s-", label="Random")
        ax.set_xlabel("Number of labeled samples")
        ax.set_ylabel("Test accuracy")
        ax.set_title("Active learning: uncertainty vs random")
        ax.legend()
        ax.grid(True, alpha=0.3)
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {args.save}")
    print("Done.")


if __name__ == "__main__":
    main()
