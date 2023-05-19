import torch
import torch.nn as nn

from .wn import WN


class ContentEncoder(nn.Module):
    def __init__(self,
            input_channels=513,
            output_channels=192,
            internal_channels=192,
            bottleneck_channels=4,
            speaker_encoding_channels=256,
            kernel_size=5,
            dilation_rate=1,
            num_resblock=4):
        super().__init__()

        self.input_layer = nn.Conv1d(input_channels, internal_channels, 1, 1, 0)
        self.wn = WN(internal_channels, kernel_size, dilation_rate, num_resblock, speaker_encoding_channels, with_speaker=False)
        self.output_layer = nn.Sequential(
                nn.Conv1d(internal_channels, bottleneck_channels, 1, 1, 0),
                nn.Conv1d(bottleneck_channels, output_channels*2, 1, 1, 0))

    def forward(self, x):
        x = self.input_layer(x)
        x = self.wn(x)
        x = self.output_layer(x)
        mu, sigma = x.chunk(2, dim=1)
        return mu, sigma

