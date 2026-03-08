# -*- coding: utf-8 -*-
import numpy as np

from nwf.vision import ConvVAEEncoder, PretrainedVisionEncoder


def test_conv_vae_encoder() -> None:
    np.random.seed(42)
    X = np.random.rand(64, 3, 32, 32).astype(np.float32)
    enc = ConvVAEEncoder(input_shape=(3, 32, 32), latent_dim=16)
    enc.fit(X, epochs=2, batch_size=32)
    z, sigma = enc.encode(X[:4])
    assert z.shape == (4, 16)
    assert sigma.shape == (4, 16)
    assert np.all(sigma > 0)


def test_conv_vae_encoder_nhwc() -> None:
    np.random.seed(42)
    X = np.random.rand(32, 32, 32, 3).astype(np.float32)
    enc = ConvVAEEncoder(input_shape=(3, 32, 32), latent_dim=8)
    enc.fit(X, epochs=1, batch_size=16)
    z, sigma = enc.encode(X[:2])
    assert z.shape == (2, 8)


def test_pretrained_vision_encoder() -> None:
    np.random.seed(42)
    X = np.random.rand(16, 3, 224, 224).astype(np.float32)
    enc = PretrainedVisionEncoder(
        backbone="resnet18", latent_dim=32, pretrained=False, trainable=False
    )
    enc.fit(X, epochs=1, batch_size=8)
    z, sigma = enc.encode(X[:2])
    assert z.shape == (2, 32)
    assert sigma.shape == (2, 32)
