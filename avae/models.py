import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from avae.base import AbstractAffinityVAE

from .base import SpatialDims


def set_layer_dim(
    ndim: SpatialDims | int,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    if ndim == SpatialDims.TWO:
        return nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d
    elif ndim == SpatialDims.THREE:
        return nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d
    elif ndim == SpatialDims.ONE:
        return nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d     # added the 1D convolutional models
    else:
        logging.error("Data must be 1D, 2D or 3D.")
        exit(1)


def dims_after_pooling(start: int, n_pools: int) -> int:
    """Calculate the size of a layer after n pooling ops.

    Parameters
    ----------
    start: int
        The size of the layer before pooling.
    n_pools: int
        The number of pooling operations.

    Returns
    -------
    int
        The size of the layer after pooling.


    """
    return start // (2**n_pools)

# EP added the function below to handle calculation of the shape of 1D convolutional layers after pooling

def dims_after_pooling_1D(length: int, n_pools: int) -> int:
    """Calculate the size of a 1D layer after n pooling ops.

    Parameters
    ----------
    start: int
        The size of the layer before pooling.
    n_pools: int
        The number of pooling operations.

    Returns
    -------
    int
        The size of the layer after pooling.


    """
    kernel_size=3
    stride=2
    padding=1

    length = ((length + (2 * padding) - (kernel_size - 1) - 1) / stride) + 1
    n_pools = n_pools - 1
    
    if n_pools > 0:
        length = dims_after_pooling_1D(length, n_pools)
    
    return int(length)


def set_device(gpu: bool) -> torch.device:
    """Set the torch device to use for training and inference.

    Parameters
    ----------
    gpu: bool
        If True, the model will be trained on GPU.

    Returns
    -------
    device: torch.device

    """
    device = torch.device(
        "cuda:0" if gpu and torch.cuda.is_available() else "cpu"
    )
    if gpu and device == "cpu":
        logging.warning(
            "\n\nWARNING: no GPU available, running on CPU instead.\n"
        )
    return device


#
# Concrete implementation of the AffinityVAE
class AffinityVAE(AbstractAffinityVAE):
    def __init__(self, encoder, decoder):
        super(AffinityVAE, self).__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.pose != self.decoder.pose:
            logging.error("Encoder and decoder pose must be the same.")
            raise RuntimeError("Encoder and decoder pose must be the same.")

        self.pose = self.encoder.pose

    def forward(self, x):
        if self.pose:
            latent_mu, latent_logvar, latent_pose = self.encoder(x)
        else:
            latent_mu, latent_logvar = self.encoder(x)
            latent_pose = None
            # reparametrise
        latent = self.reparametrise(latent_mu, latent_logvar)
        # decode
        x_recon = self.decoder(latent, latent_pose)  # pose set to None if pd=0
        return x_recon, latent_mu, latent_logvar, latent, latent_pose
        # encode

    def reparametrise(self, mu, log_var):        # output of this function, a vector of length n_latent dims if input to the decoder
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu
