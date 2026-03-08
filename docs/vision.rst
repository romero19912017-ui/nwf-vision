Vision Module
=============

nwf-vision provides encoders for computer vision tasks with NWF.

Encoders
--------

.. automodule:: nwf.vision.encoders
   :members:
   :undoc-members:
   :show-inheritance:

ConvVAEEncoder
~~~~~~~~~~~~~~

Convolutional VAE for images (CIFAR 32x32, MNIST). Produces (z, sigma) for NWF charges.

.. code-block:: python

   from nwf.vision import ConvVAEEncoder
   from nwf import Charge, Field

   enc = ConvVAEEncoder(input_shape=(3, 32, 32), latent_dim=64)
   enc.fit(images, epochs=20)
   z, sigma = enc.encode(images[:10])
   for i in range(10):
       field.add(Charge(z=z[i], sigma=sigma[i]), labels=[labels[i]])

PretrainedVisionEncoder
~~~~~~~~~~~~~~~~~~~~~~~

ResNet backbone with (z, sigma) head. Uses pretrained torchvision weights.

.. code-block:: python

   from nwf.vision import PretrainedVisionEncoder

   enc = PretrainedVisionEncoder(
       backbone="resnet18",
       latent_dim=64,
       pretrained=True,
       trainable=False,
   )
   enc.fit(images, epochs=5)
   z, sigma = enc.encode(images)

Examples
--------

Split-CIFAR-10
~~~~~~~~~~~~~~

Incremental learning: 5 tasks, 2 classes each. Train ConvVAE once, add charges per task.

.. code-block:: bash

   python examples/split_cifar.py --epochs 5 --n-tasks 5
