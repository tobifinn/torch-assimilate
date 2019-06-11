Variational inference toolbox
=============================
This variational inference toolbox can be used in conjunction with a data
assimilation algorithm. The main purpose of this toolbox is to simplify training
of neural networks for data assimilation. For variational inference,
data assimilation is recasted into a variational autoencoder framework
(:py:class:``pytassim.toolbox.autoencoder.Autoencoder``), which is also the main
class in this implementation. For implicit latent states, Generative
adversarial networks (GANs) are used. In these GANs, a discriminator is trained
to classify between real and fake samples. This adversarial training is
supported and prepared by the discriminator classes
(:py:class:``pytassim.toolbox.discriminator.standard.StandardDisc``). Normal
pytorch loss-functions can be wrapped and used for autoencoding with the
:py:class:``pytassim.toolbox.loss.LossWrapper``. We also implemented a new
optimizer (:py:class:``pytassim.toolbox.heun.HeunMethod``), which can be used in
PyTorch. This new optimizer is based on Heun's method and is an implementation
of an averaged stochastic gradient descent algorithm for training.


Variational autoencoder
-----------------------

.. autosummary::
    pytassim.toolbox.autoencoder.Autoencoder


Discriminators
--------------

Standard discriminator
^^^^^^^^^^^^^^^^^^^^^^


.. autosummary::
    pytassim.toolbox.discriminator.standard.StandardDisc


Regularizations
---------------

Gradient penalty
^^^^^^^^^^^^^^^^

.. autosummary::
    pytassim.toolbox.gradient_penalty.zero_grad_penalty


Utilities
---------

Loss wrapper
^^^^^^^^^^^^


.. autosummary::
    pytassim.toolbox.loss.LossWrapper
    pytassim.toolbox.heun.HeunMethod
