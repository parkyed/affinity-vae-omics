# Concrete implementation of Decoder
import logging

import numpy as np
import torch
import torch.nn as nn # add this as required for DecoderC - could find a way to add back to models.py

from avae.decoders.base import AbstractDecoder
from avae.models import dims_after_pooling, set_layer_dim, dims_after_pooling_1D  # added dims_after_pooling_1D


class Decoder(AbstractDecoder):
    """Affinity decoder. Includes optional pose component merge.

    Parameters
    ----------
    input_size: tuple (X, Y) or tuple (X, Y, Z)
        Tuple representing the size of the data for each image
        dimension X, Y and Z.
    latent_dims: int
        Number of bottleneck latent dimensions.
    pose_dims : int
        Number of bottleneck pose dimensions.
    capacity : int (None)
        The capacity of the network - initial number of nodes doubled at each
        depth.
    depth : int (None)
        The depth of the network - number of downsampling layers.
    filters : list : [int] (None)
        List of filter sizes, where len(filters) becomes network depth.
    bnorm : bool (True)
        If True, turns BatchNormalisation on.
    """

    def __init__(
        self,
        input_size: tuple,
        capacity: int | None = None,
        depth: int = 4,
        latent_dims: int = 8,
        pose_dims: int = 0,
        filters: list[int] | None = None,
        bnorm: bool = True,
    ):

        super(Decoder, self).__init__()
        self.filters = []
        if capacity is None and filters is None:
            raise RuntimeError(
                "Pass either capacity or filters when definining avae.Decoder."
            )
        elif filters is not None and len(filters) != 0:
            if 0 in filters:
                raise RuntimeError("Filter list cannot contain zeros.")
            self.filters = filters
            if depth is not None:
                logging.warning(
                    "You've passed 'filters' parameter as well as 'depth'. Filters take"
                    " priority so 'depth' and 'capacity' will be disregarded."
                )
        elif capacity is not None:
            if depth is None:
                raise RuntimeError(
                    "When passing initial 'capacity' parameter in avae.Encoder,"
                    " provide 'depth' parameter too."
                )
            self.filters = [capacity * 2**x for x in range(depth)]
        else:
            raise RuntimeError(
                "You must provide either capacity or filters when definity ave.Decoder."
            )

        assert all(
            [int(x) == x for x in np.array(input_size) / (2**depth)]
        ), (
            "Input size not compatible with --depth. Input must be divisible "
            "by {}.".format(2**depth)
        )

        self.bottom_dim = tuple(
            [int(i / (2 ** len(self.filters))) for i in input_size]
        )
        self.pose = not (pose_dims == 0)
        self.bnorm = bnorm

        # define layer dimensions
        CONV, TCONV, BNORM = set_layer_dim(len(input_size))

        #  iteratively define deconvolution and batch normalisation layers
        self.conv_dec = torch.nn.ModuleList()
        if self.bnorm:
            self.norm_dec = torch.nn.ModuleList()

        for d in reversed(range(len(self.filters))):
            self.conv_dec.append(
                TCONV(
                    in_channels=self.filters[d],
                    out_channels=(self.filters[d - 1] if d != 0 else 1),
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            if self.bnorm and d != 0:
                self.norm_dec.append(BNORM(self.filters[d - 1]))

        # define fully connected layers
        self.ch = (
            1 if depth == 0 else self.filters[-1]
        )  # allow for no convolutions
        if self.pose:
            self.fc = torch.nn.Linear(
                in_features=pose_dims + latent_dims,
                out_features=self.ch * np.prod(self.bottom_dim),
            )
        else:
            self.fc = torch.nn.Linear(
                in_features=latent_dims,
                out_features=self.ch * np.prod(self.bottom_dim),
            )

    def forward(self, x, x_pose):
        """Decoder forward pass.

        Parameters
        ----------
        x : torch.Tensor (N, latent_dims)
            Mini-batch of reparametrised encoder outputs, where N stands for
            the number of samples in the mini-batch and
            'latent_dims' defines the number of latent dimensions.
        x_pose : torch.Tensor (N, pose_dims)
            Mini-batch of outputs representing pose capturing the within-class
            variance, where N stands for the number
            of samples in the mini-batch and 'pose_dims' defines the number of
            pose dimensions.

        Returns
        -------
        x : torch.Tensor (N, CH, Z, Y, X)
            Mini-batch of outputs, where N stands for the number of samples in
            the mini-batch, CH stands for number of
            channels and X, Y, Z define input dimensions.

        """
        if self.pose:
            x = self.fc(torch.cat((x, x_pose), -1))
        else:
            x = self.fc(x)
        x = x.view(x.size(0), self.ch, *self.bottom_dim)
        for d in range(len(self.filters) - 1):
            if self.bnorm:
                x = self.norm_dec[d](
                    torch.nn.functional.relu(self.conv_dec[d](x))
                )
            else:
                x = torch.nn.functional.relu(self.conv_dec[d](x))
        x = torch.sigmoid(self.conv_dec[-1](x))
        return x


class DecoderA(AbstractDecoder):
    def __init__(
        self,
        input_size: tuple,
        capacity: int = 8,
        depth: int = 4,
        latent_dims: int = 8,
        pose_dims: int = 0,
        bnorm: bool = False,
    ):
        super(DecoderA, self).__init__()
        self.pose = not (pose_dims == 0)
        self.bnorm = bnorm

        if len(input_size) == 1: # added a if else to specify the correct condition for a 1D input (assume fixed kernal size, stride and padding)
            assert int(dims_after_pooling_1D(int(np.array(input_size)[0]), depth)) == dims_after_pooling_1D(int(np.array(input_size)[0]), depth), "Input length not compatible with --depth, kernal, stide and padding"
        else:
            assert all(
                [int(x) == x for x in np.array(input_size) / (2**depth)]
            ), (
                "Input size not compatible with --depth. Input must be divisible "
                "by {}.".format(2**depth)
            )
        filters = [capacity * 2**x for x in range(depth)]

        if len(input_size) == 1:                  # added if else - output shape different for 1D vs. 2D inputs. dims_after_pooling_1D in models.py
            unflat_shape = tuple([filters[-1],
                                 dims_after_pooling_1D(int(np.array(input_size)[0]), depth)])
        else:
            unflat_shape = tuple(
                [
                    filters[-1],
                ]
                + [dims_after_pooling(ax, depth) for ax in input_size]
            )
        flat_shape = np.prod(unflat_shape)

        ndim = len(unflat_shape[1:])

        _, conv_T, BNORM = set_layer_dim(ndim)

        self.decoder = torch.nn.Sequential()

        self.decoder.append(
            torch.nn.Linear(latent_dims + pose_dims, flat_shape)
        )
        self.decoder.append(torch.nn.Unflatten(-1, unflat_shape))

        if len(input_size) == 1: # EP added this if else as the decoder doesn't produce a 1d vector of the input size - kernal/stride/padding different to encoder
            for d in reversed(range(len(filters))):
                if d != 0:
                    if not d == len(filters) - 1:
                        self.decoder.append(
                            conv_T(
                                in_channels=filters[d],
                                out_channels=filters[d - 1],
                                kernel_size=3,
                                stride=2,
                                padding=1,
                            )
                        )
                    else:
                        self.decoder.append(
                            conv_T(
                                in_channels=filters[d],
                                out_channels=filters[d - 1],
                                kernel_size=3,
                                stride=2,
                                padding=1,  # don't understand why the padding is set to zero for the first layer
                            )
                        )
                    if self.bnorm:
                        self.decoder.append(BNORM(filters[d - 1]))
                    self.decoder.append(torch.nn.ReLU(True))

            self.decoder.append(
                conv_T(
                    in_channels=filters[0],
                    out_channels=1,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
        else:
            for d in reversed(range(len(filters))):
                if d != 0:
                    if not d == len(filters) - 1:
                        self.decoder.append(
                            conv_T(
                                in_channels=filters[d],
                                out_channels=filters[d - 1],
                                kernel_size=3,
                                stride=2,
                                padding=1,
                            )
                        )
                    else:
                        self.decoder.append(
                            conv_T(
                                in_channels=filters[d],
                                out_channels=filters[d - 1],
                                kernel_size=3,
                                stride=2,
                                padding=0,     # don't understand why the padding is set to zero for the first layer
                            )
                        )
                    if self.bnorm:
                        self.decoder.append(BNORM(filters[d - 1]))
                    self.decoder.append(torch.nn.ReLU(True))

            self.decoder.append(
                conv_T(
                    in_channels=filters[0],
                    out_channels=1,
                    kernel_size=2,
                    stride=2,
                    padding=1,
                )
            )

    def forward(self, x, x_pose):
        if self.pose:
            return self.decoder(torch.cat([x_pose, x], dim=-1))
        else:
            return self.decoder(x)


class DecoderB(AbstractDecoder):
    """Affinity decoder. Includes optional pose component merge.

    Parameters
    ----------
    capacity : int
        The capacity of the network - initial number of nodes doubled at each
        depth.
    depth : int
        The depth of the network - number of downsampling layers.
    latent_dims: int
        Number of bottleneck latent dimensions.
    pose_dims : int
        Number of bottleneck pose dimensions.
    """

    def __init__(
        self,
        input_size: tuple,
        capacity: int,
        depth: int = 4,
        latent_dims: int = 8,
        pose_dims: int = 0,
    ):
        super(DecoderB, self).__init__()
        self.capacity = capacity
        self.depth = depth
        self.pose = not (pose_dims == 0)

        assert all(
            [int(x) == x for x in np.array(input_size) / (2**depth)]
        ), (
            "Input size not compatible with --depth. Input must be divisible "
            "by {}.".format(2**depth)
        )
        _, TCONV, BNORM = set_layer_dim(len(input_size))
        self.bottom_dim = tuple([int(i / (2**depth)) for i in input_size])

        #  iteratively define deconvolution and batch normalisation layers
        self.conv_dec = torch.nn.ModuleList()
        self.norm_dec = torch.nn.ModuleList()
        prev_sh = self.capacity * depth
        for d in range(depth, 0, -1):
            sh = self.capacity * (d - 1) if d != 1 else 1
            self.conv_dec.append(
                TCONV(
                    in_channels=prev_sh,
                    out_channels=sh,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            self.norm_dec.append(BNORM(sh))
            prev_sh = sh

        # define fully connected layers
        self.chf = (
            1 if depth == 0 else self.capacity * depth
        )  # allow for no convolutions
        if self.pose:
            self.fc = torch.nn.Linear(
                in_features=pose_dims + latent_dims,
                out_features=self.chf * np.prod(self.bottom_dim),
            )
        else:
            self.fc = torch.nn.Linear(
                in_features=latent_dims,
                out_features=self.chf * np.prod(self.bottom_dim),
            )

    def forward(self, x, x_pose):
        """Decoder forward pass.

        Parameters
        ----------
        x : torch.Tensor (N, latent_dims)
            Mini-batch of reparametrised encoder outputs, where N stands for
            the number of samples in the mini-batch and
            'latent_dims' defines the number of latent dimensions.
        x_pose : torch.Tensor (N, pose_dims)
            Mini-batch of outputs representing pose capturing the within-class
            variance, where N stands for the number
            of samples in the mini-batch and 'pose_dims' defines the number of
            pose dimensions.

        Returns
        -------
        x : torch.Tensor (N, CH, Z, Y, X)
            Mini-batch of outputs, where N stands for the number of samples in
            the mini-batch, CH stands for number of
            channels and X, Y, Z define input dimensions.

        """
        if self.pose:
            x = self.fc(torch.cat((x, x_pose), -1))
        else:
            x = self.fc(x)
        x = x.view(x.size(0), self.chf, *self.bottom_dim)
        for d in range(self.depth - 1):
            x = self.norm_dec[d](torch.nn.functional.relu(self.conv_dec[d](x)))
        x = torch.sigmoid(self.conv_dec[-1](x))
        return x


class DecoderC(AbstractDecoder):
    def __init__(
            self,
            input_size: tuple,
            capacity: int = 8, # reuse capacity (number of channels/ filters in CNN) as the number of nodes in each hidden layer
            depth: int = 2,    # reuse depth as the number of fully connected hidden layers
            latent_dims: int = 8,
            pose_dims: int = 0,
            bnorm: bool = True,
    ):
        super(DecoderC, self).__init__()
        self.pose = not (pose_dims == 0)
        self.bnorm = bnorm

        assert len(input_size) == 1, "Input must be a 1D vector for this Encoder / Decoder"

        n_hidden = np.repeat(capacity, depth)  # define n_hidden as a function of capacity and depth

        self.decoder = torch.nn.Sequential()

        input_features = int(np.array(input_size)[0])  # the dimensions of the input

        self.decoder.append(torch.nn.Linear(latent_dims + pose_dims, n_hidden[-1]))

        for d in reversed(
                range(len(n_hidden))):  # recursive adding of hidden layers, based on the n_hidden in each layer
            self.decoder.append(
                nn.Linear(
                    in_features=n_hidden[d],
                    out_features=n_hidden[d-1]
                )
            )
            self.decoder.append(torch.nn.ReLU(True))
            # input_features = n_hidden[d-1]  # update the number of nodes in the hidden layer to input to the next layer

        self.decoder.append(
            torch.nn.Linear(n_hidden[0], input_features)
        )

    def forward(self, x, x_pose):
        if self.pose:
            return self.decoder(torch.cat([x_pose, x], dim=-1))
        else:
            return self.decoder(x)
