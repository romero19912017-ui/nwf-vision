# -*- coding: utf-8 -*-
"""Split-CIFAR-10: incremental learning with NWF.

Splits CIFAR-10 into 5 tasks (2 classes each). Trains ConvVAE on all data,
then adds charges per task to Field. Evaluates accuracy after each task.
"""
from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from nwf import Charge, Field
from nwf.vision import ConvVAEEncoder


def get_cifar10(root: str = "./data") -> Tuple[np.ndarray, np.ndarray]:
    ds = CIFAR10(root=root, train=True, download=True, transform=ToTensor())
    X = np.stack([ds[i][0].numpy() for i in range(len(ds))])
    y = np.array([ds[i][1] for i in range(len(ds))])
    return X, y


def split_tasks(
    n_tasks: int = 5, n_classes: int = 10
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Split data into tasks. Each task: 2 classes for CIFAR-10."""
    X, y = get_cifar10()
    tasks = []
    classes_per_task = n_classes // n_tasks
    for t in range(n_tasks):
        c_start = t * classes_per_task
        c_end = (t + 1) * classes_per_task
        mask = (y >= c_start) & (y < c_end)
        X_t = X[mask]
        y_t = y[mask]
        y_local = y_t - c_start
        tasks.append((X_t, y_t, y_local, np.arange(c_start, c_end)))
    return tasks


def compute_class_charges(
    enc: ConvVAEEncoder,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean z and std sigma per class."""
    z_all, sigma_all = enc.encode(X)
    if z_all.ndim == 1:
        z_all = z_all.reshape(1, -1)
        sigma_all = sigma_all.reshape(1, -1)
    classes = np.unique(y)
    z_means = []
    sigma_means = []
    for c in classes:
        mask = y == c
        z_c = z_all[mask]
        s_c = sigma_all[mask]
        z_means.append(np.mean(z_c, axis=0))
        sigma_means.append(np.mean(s_c, axis=0) + 1e-6)
    return np.array(z_means), np.array(sigma_means)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--n-tasks", type=int, default=5)
    args = parser.parse_args()

    tasks = split_tasks(n_tasks=args.n_tasks)
    X_full = np.concatenate([t[0] for t in tasks])
    y_full = np.concatenate([t[1] for t in tasks])

    print("Training ConvVAE on all data...")
    enc = ConvVAEEncoder(input_shape=(3, 32, 32), latent_dim=args.latent_dim)
    enc.fit(X_full, epochs=args.epochs, batch_size=128)

    field = Field()
    accuracies = []

    for task_id, (X_t, y_t, _, class_ids) in enumerate(tasks):
        z_mean, sigma_mean = compute_class_charges(enc, X_t, y_t)
        for i, c in enumerate(class_ids):
            ch = Charge(z=z_mean[i].astype(np.float64), sigma=sigma_mean[i].astype(np.float64))
            field.add(ch, labels=[int(c)], ids=[task_id * 10 + c])
        dist, idx, labels = field.search(
            Charge(z=z_mean[0], sigma=sigma_mean[0]), k=1
        )
        n_correct = 0
        n_total = 0
        for t in range(task_id + 1):
            X_te, y_te = tasks[t][0], tasks[t][1]
            z_te, s_te = enc.encode(X_te[:100])
            if z_te.ndim == 1:
                z_te = z_te.reshape(1, -1)
                s_te = s_te.reshape(1, -1)
            for i in range(len(z_te)):
                q = Charge(z=z_te[i], sigma=s_te[i])
                _, _, lab = field.search(q, k=1)
                pred = lab[0][0]
                if pred == y_te[i]:
                    n_correct += 1
                n_total += 1
        acc = n_correct / n_total if n_total > 0 else 0
        accuracies.append(acc)
        print(f"After task {task_id + 1}: accuracy = {acc:.4f}")

    print(f"Final avg accuracy: {np.mean(accuracies):.4f}")


if __name__ == "__main__":
    main()
