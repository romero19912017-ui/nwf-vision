# -*- coding: utf-8 -*-
"""Vision encoders for NWF: ConvVAE, Pretrained (ResNet, etc)."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

import torch
import torch.nn as nn



class _ConvVAE(nn.Module):
    """Convolutional VAE for images (e.g. CIFAR 32x32). 32->16->8->4."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 64,
        hidden_dims: Tuple[int, ...] = (32, 64, 128),
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self._hidden_dims = hidden_dims
        # Encoder: 32x32 -> 16 -> 8 -> 4
        enc = []
        d = in_channels
        for h in hidden_dims:
            enc += [
                nn.Conv2d(d, h, 3, stride=2, padding=1),
                nn.BatchNorm2d(h),
                nn.ReLU(),
            ]
            d = h
        self.encoder = nn.Sequential(*enc)
        self.enc_dim = d * 4 * 4
        self.fc_mu = nn.Linear(self.enc_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_dim, latent_dim)
        # Decoder: 4x4 -> 8 -> 16 -> 32
        self.fc_dec = nn.Linear(latent_dim, self.enc_dim)
        dec = []
        rev_dims = [in_channels] + list(hidden_dims)
        for i in range(len(hidden_dims), 0, -1):
            cin, cout = rev_dims[i], rev_dims[i - 1]
            dec.append(nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1))
            if i > 1:
                dec.append(nn.BatchNorm2d(cout))
                dec.append(nn.ReLU())
            else:
                dec.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.reshape(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        h = h.reshape(z.size(0), self._hidden_dims[-1], 4, 4)
        return self.decoder(h)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = self.decode(z)
        return recon, mu, logvar


class ConvVAEEncoder:
    """Convolutional VAE producing (z, sigma) for NWF charges."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
        latent_dim: int = 64,
        hidden_dims: Tuple[int, ...] = (32, 64, 128),
        device: Optional[str] = None,
    ) -> None:
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.device = device or "cpu"
        in_channels = input_shape[0]
        self._model = _ConvVAE(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)

    def fit(
        self,
        train_data: Union[np.ndarray, torch.Tensor],
        epochs: int = 20,
        batch_size: int = 128,
        lr: float = 1e-3,
    ) -> "ConvVAEEncoder":
        """Train ConvVAE on images. train_data: (N, C, H, W) or (N, H, W, C)."""
        self._model.train()
        if isinstance(train_data, np.ndarray):
            X = torch.FloatTensor(train_data).to(self.device)
        else:
            X = train_data.to(self.device)
        if X.dim() == 3:
            X = X.unsqueeze(0)
        if X.shape[-1] == self.input_shape[0]:
            X = X.permute(0, 3, 1, 2)
        n = len(X)
        opt = torch.optim.Adam(self._model.parameters(), lr=lr)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                batch = X[idx]
                recon, mu, logvar = self._model(batch)
                bce = nn.functional.binary_cross_entropy(
                    recon, batch, reduction="sum"
                )
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = (bce + kl) / batch.size(0)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._model.eval()
        return self

    def encode(self, x: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """Encode to (z, sigma). Returns numpy arrays."""
        self._model.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                t = torch.FloatTensor(x).to(self.device)
            else:
                t = x.to(self.device)
            if t.dim() == 3:
                t = t.unsqueeze(0)
            if t.shape[1] != self.input_shape[0] and t.shape[-1] == self.input_shape[0]:
                t = t.permute(0, 3, 1, 2)
            mu, logvar = self._model.encode(t)
            sigma = torch.exp(0.5 * logvar)
            z = mu.cpu().numpy()
            s = sigma.cpu().numpy()
        return z.squeeze(), s.squeeze()


class _ResNetHead(nn.Module):
    """Head producing (z, log_sigma) from backbone features."""

    def __init__(self, in_features: int, latent_dim: int) -> None:
        super().__init__()
        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_logvar = nn.Linear(in_features, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fc_mu(x), self.fc_logvar(x)


class PretrainedVisionEncoder:
    """Pretrained ResNet with (z, sigma) head for NWF."""

    def __init__(
        self,
        backbone: str = "resnet18",
        latent_dim: int = 64,
        pretrained: bool = True,
        trainable: bool = True,
        device: Optional[str] = None,
    ) -> None:
        import torchvision.models as models

        self.backbone_name = backbone
        self.latent_dim = latent_dim
        self.device = device or "cpu"
        if backbone == "resnet18":
            m = models.resnet18(weights="DEFAULT" if pretrained else None)
            in_features = m.fc.in_features
            m.fc = nn.Identity()
        elif backbone == "resnet34":
            m = models.resnet34(weights="DEFAULT" if pretrained else None)
            in_features = m.fc.in_features
            m.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        self._backbone = m.to(self.device)
        self._head = _ResNetHead(in_features, latent_dim).to(self.device)
        if not trainable:
            for p in self._backbone.parameters():
                p.requires_grad = False
        self._in_features = in_features

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self._backbone(x)
        return out.reshape(out.size(0), -1)

    def fit(
        self,
        train_data: Union[np.ndarray, torch.Tensor],
        train_labels: Optional[np.ndarray] = None,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
    ) -> "PretrainedVisionEncoder":
        """Train head on features (unsupervised: maximize ELBO of unit Gaussian prior)."""
        trainable = [p for p in self._head.parameters() if p.requires_grad]
        trainable += [p for p in self._backbone.parameters() if p.requires_grad]
        opt = torch.optim.Adam(trainable, lr=lr)
        if isinstance(train_data, np.ndarray):
            X = torch.FloatTensor(train_data).to(self.device)
        else:
            X = train_data.to(self.device)
        if X.dim() == 3:
            X = X.unsqueeze(0)
        n = len(X)
        self._backbone.train()
        self._head.train()
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                batch = X[idx]
                feat = self._get_features(batch)
                mu, logvar = self._head(feat)
                loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._backbone.eval()
        self._head.eval()
        return self

    def encode(self, x: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """Encode to (z, sigma)."""
        self._backbone.eval()
        self._head.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                t = torch.FloatTensor(x).to(self.device)
            else:
                t = x.to(self.device)
            if t.dim() == 3:
                t = t.unsqueeze(0)
            feat = self._get_features(t)
            mu, logvar = self._head(feat)
            sigma = torch.exp(0.5 * logvar)
            z = mu.cpu().numpy()
            s = sigma.cpu().numpy()
        return z.squeeze(), s.squeeze()
