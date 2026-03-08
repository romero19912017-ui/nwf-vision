# nwf-vision

[![PyPI version](https://badge.fury.io/py/nwf-vision.svg)](https://pypi.org/project/nwf-vision/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/romero19912017-ui/nwf-vision/actions/workflows/test.yml/badge.svg)](https://github.com/romero19912017-ui/nwf-vision/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## NWF for Computer Vision

`nwf-vision` provides encoders and examples for **incremental learning**, **OOD detection**, and **active learning** on images. Built on top of `nwf-core`, it uses its indices, metrics, and calibrators.

### Features

- **ConvVAEEncoder** — convolutional VAE for images (CIFAR 32x32, MNIST)
- **PretrainedVisionEncoder** — ResNet18/34 with (z, sigma) head for NWF charges
- **Split-CIFAR-10 example** — incremental classification without catastrophic forgetting
- **OOD detection example** — CIFAR-10 (in) vs SVHN (out) using semantic potential
- **Active learning example** — uncertainty sampling vs random on CIFAR-10
- Support for NHWC and NCHW input layouts

---

## Installation

```bash
pip install nwf-vision
```

Requires: `nwf-core`, `torch`, `torchvision`.

---

## Encoders

### ConvVAEEncoder
Convolutional VAE producing `(z, sigma)` for NWF charges. Suitable for small images (32x32).

```python
from nwf.vision import ConvVAEEncoder
import numpy as np

enc = ConvVAEEncoder(input_shape=(3, 32, 32), latent_dim=64)
enc.fit(train_images, epochs=20)
z, sigma = enc.encode(images[:10])
```

- `input_shape` — (C, H, W), e.g. (3, 32, 32) for CIFAR
- `latent_dim` — dimension of z
- `hidden_dims` — encoder channel sizes (default: 32, 64, 128)
- `fit(train_data, epochs, batch_size, lr)` — train VAE
- `encode(x)` — returns `(z, sigma)` as numpy arrays; accepts NHWC or NCHW

### PretrainedVisionEncoder
ResNet backbone from torchvision with a head producing `(z, sigma)`.

```python
from nwf.vision import PretrainedVisionEncoder

enc = PretrainedVisionEncoder(
    backbone="resnet18",
    latent_dim=64,
    pretrained=True,
    trainable=False,
)
enc.fit(images, epochs=5)
z, sigma = enc.encode(images)
```

- `backbone` — "resnet18" or "resnet34"
- `pretrained` — use ImageNet weights
- `trainable` — fine-tune backbone or freeze it
- Input: (N, 3, 224, 224) for ImageNet-style models

---

## Example: Split-CIFAR-10

Incremental learning: 5 tasks, 2 classes each. Train ConvVAE once, add charges per task.

```bash
pip install nwf-core nwf-vision
python examples/split_cifar.py --epochs 5 --n-tasks 5
```

The example demonstrates adding new classes without retraining the encoder and without catastrophic forgetting.

---

## Examples: OOD and Active Learning

```bash
python examples/ood_detection.py      # CIFAR-10 vs SVHN, AUROC
python examples/active_learning.py    # Uncertainty vs random sampling
```

---

## Links

- **Article (Habr):** [Нейровесовые Поля (NWF)](https://github.com/romero19912017-ui/nwf-research/blob/main/HABR.md) — теория, continual learning
- **Core:** [nwf-core](https://github.com/romero19912017-ui/nwf-core)
- **Research:** [nwf-research](https://github.com/romero19912017-ui/nwf-research)

## License

MIT

---

# nwf-vision (Русский)

## NWF для компьютерного зрения

`nwf-vision` предоставляет энкодеры и примеры для **инкрементального обучения**, **OOD-детекции** и **активного обучения** на изображениях.

### Компоненты

- **ConvVAEEncoder** — свёрточный VAE для изображений (CIFAR 32x32, MNIST); выход `(z, sigma)`
- **PretrainedVisionEncoder** — ResNet18/34 с головой для (z, sigma); предобученные веса
- **Split-CIFAR-10** — пример инкрементальной классификации без катастрофического забывания
- **OOD detection** — пример CIFAR-10 vs SVHN через потенциал
- **Active learning** — uncertainty sampling vs random

### Установка

```bash
pip install nwf-vision
```

### Пример

```python
from nwf.vision import ConvVAEEncoder
from nwf import Charge, Field

enc = ConvVAEEncoder(input_shape=(3, 32, 32), latent_dim=64)
enc.fit(images, epochs=10)
z, sigma = enc.encode(images[:5])

field = Field()
for i in range(5):
    field.add(Charge(z=z[i], sigma=sigma[i]), labels=[labels[i]])
```

### Лицензия

MIT
