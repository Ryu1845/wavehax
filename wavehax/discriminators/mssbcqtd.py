# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from einops import rearrange
import torchaudio.transforms as T

from nnAudio import features

from .univnet import MultiPeriodDiscriminator, MultiResolutionDiscriminator

LRELU_SLOPE = 0.1


class DiscriminatorCQT(nn.Module):
    def __init__(
        self, 
        filters: int,
        max_filters: int,
        filters_scale: int,
        dilations: list[int],
        in_channels: int,
        out_channels: int,
        sample_rate: int,
        hop_length: int, 
        n_octaves: int, 
        bins_per_octave: int,
    ):
        super(DiscriminatorCQT, self).__init__()
        self.filters = filters
        self.max_filters = max_filters
        self.filters_scale = filters_scale
        self.kernel_size = (3, 9)
        self.dilations = dilations
        self.stride = (1, 2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fs = sample_rate
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave

        self.cqt_transform = features.cqt.CQT2010v2(
            sr=self.fs * 2,
            hop_length=self.hop_length,
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )

        self.conv_pres = nn.ModuleList()
        for i in range(self.n_octaves):
            self.conv_pres.append(
                nn.Conv2d(
                    self.in_channels * 2,
                    self.in_channels * 2,
                    kernel_size=self.kernel_size,
                    padding=get_2d_padding(self.kernel_size),
                )
            )

        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Conv2d(
                self.in_channels * 2,
                self.filters,
                kernel_size=self.kernel_size,
                padding=get_2d_padding(self.kernel_size),
            )
        )

        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for i, dilation in enumerate(self.dilations):
            out_chs = min(
                (self.filters_scale ** (i + 1)) * self.filters, self.max_filters
            )
            self.convs.append(
                weight_norm(nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    dilation=(dilation, 1),
                    padding=get_2d_padding(self.kernel_size, (dilation, 1)),
                ))
            )
            in_chs = out_chs
        out_chs = min(
            (self.filters_scale ** (len(self.dilations) + 1)) * self.filters,
            self.max_filters,
        )
        self.convs.append(
            weight_norm(nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                padding=get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
            ))
        )

        self.conv_post = weight_norm(nn.Conv2d(
            out_chs,
            self.out_channels,
            kernel_size=(self.kernel_size[0], self.kernel_size[0]),
            padding=get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
        ))

        self.activation = torch.nn.LeakyReLU(negative_slope=LRELU_SLOPE)
        self.resample = T.Resample(orig_freq=self.fs, new_freq=self.fs * 2)

    def forward(self, x):
        fmap = []

        x = self.resample(x)

        z = self.cqt_transform(x)

        z_amplitude = z[:, :, :, 0].unsqueeze(1)
        z_phase = z[:, :, :, 1].unsqueeze(1)

        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = rearrange(z, "b c w t -> b c t w")

        latent_z = []
        for i in range(self.n_octaves):
            latent_z.append(
                self.conv_pres[i](
                    z[
                        :,
                        :,
                        :,
                        i * self.bins_per_octave : (i + 1) * self.bins_per_octave,
                    ]
                )
            )
        latent_z = torch.cat(latent_z, dim=-1)

        for i, l in enumerate(self.convs):
            latent_z = l(latent_z)

            latent_z = self.activation(latent_z)
            fmap.append(latent_z)

        latent_z = self.conv_post(latent_z)

        return latent_z, fmap


class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(
        self, 
        filters: int,
        max_filters: int,
        filters_scale: int,
        dilations: list[int],
        in_channels: int,
        out_channels: int,
        sample_rate: int,
        hop_lengths: list[int], 
        n_octaves: list[int], 
        bins_per_octave: list[int],
    ):
        super(MultiScaleSubbandCQTDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorCQT(
                    filters,
                    max_filters,
                    filters_scale,
                    dilations,
                    in_channels,
                    out_channels,
                    sample_rate,
                    hop_length=hop_lengths[i],
                    n_octaves=n_octaves[i],
                    bins_per_octave=bins_per_octaves[i],
                )
                for i in range(len(hop_lengths))
            ]
        )

    def forward(self, y):
        y_d = []
        fmap = []

        for disc in self.discriminators:
            y_d, fmap_d = disc(y)
            y_d.append(y_d)
            fmap.append(fmap_d)

        return y_d, fmap


class MultiDiscriminator(nn.Module):
    """
    UnivNet's combined discriminator module + MultiScaleSubbandCQTDiscriminator.

    This module combines the multi-resolution spectral discriminator and the multi-period discriminator,
    providing a comprehensive analysis of input waveforms by considering both time-domain and frequency-domain features.
    """

    def __init__(
        self,
        # Multi-period discriminator related
        periods: List[int],
        period_discriminator_params: Dict,
        # Multi-resolution discriminator related
        fft_sizes: List[int],
        hop_sizes: List[int],
        win_lengths: List[int],
        spectral_discriminator_params: Dict,
        # Multi-subband cqt discriminator related
        cqt_discriminator_params: Dict,
    ) -> None:
        """
        Initilize the MultiResolutionMultiPeriodDiscriminator module.

        Args:
            periods (List[int]): List of periods for the HiFi-GAN period discriminators.
            period_discriminator_params (Dict): Common parameters for initializing the period discriminators.
            fft_sizes (List[int]): List of FFT sizes for the spectral discriminators.
            hop_sizes (List[int]): List of hop sizes for the spectral discriminators.
            win_lengths (List[int]): List of window lengths for the spectral discriminators.
            window (str): Name of the window function.
            spectral_discriminator_params (Dict): Common parameters for initializing the spectral discriminators.
        """
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(
            periods=periods,
            discriminator_params=period_discriminator_params,
        )
        self.mrd = MultiResolutionDiscriminator(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            discriminator_params=spectral_discriminator_params,
        )
        self.mssbcqtd = MultiScaleSubbandCQTDiscriminator(**cqt_discriminator_params)

    def forward(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Calculate forward propagation.

         Args:
            x (Tensor): Input waveforms with shape (batch, 1, length).

        Returns:
            List[Tensor]: List of outputs from each discriminator.
            List[Tensor]: List of feature maps from all discriminators.
        """
        mpd_outs, mpd_fmaps = self.mpd(x)
        mrd_outs, mrd_fmaps = self.mrd(x)
        mssbcqtd_outs, mssbcqtd_fmaps = self.mssbcqtd(x)
        outs = mpd_outs + mrd_outs + mssbcqtd_outs
        fmaps = mpd_fmaps + mrd_fmaps + mssbcqtd_fmaps

        return outs, fmaps
