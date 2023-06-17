import torch
import torch.nn as nn

from .wn import WN


class PosteriorEncoder(nn.Module):
    def __init__(self,
            input_channels=513,
            output_channels=96,
            internal_channels=96,
            speaker_encoding_channels=128,
            kernel_size=5,
            dilation_rate=1,
            num_resblock=16):
        super().__init__()

        self.input_layer = nn.Conv1d(input_channels, internal_channels, 1, 1, 0)
        self.wn = WN(internal_channels, kernel_size, dilation_rate, num_resblock, speaker_encoding_channels)
        self.output_layer = nn.Conv1d(internal_channels, output_channels*2, 1, 1, 0)

    def forward(self, x, speaker):
        x = self.input_layer(x)
        x = self.wn(x, speaker)
        x = self.output_layer(x)
        mu, sigma = x.chunk(2, dim=1)
        return mu, sigma

