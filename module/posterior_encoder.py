import torch
import torch.nn as nn

from .wn import WN


class PosteriorEncoder(nn.Module):
    def __init__(self,
            input_channels=513,
            output_channels=192,
            internal_channels=192,
            speaker_encoding_channels=256,
            kernel_size=5,
            dilation_rate=1,
            num_resblock=16):
        super().__init__()

        self.input_layer = nn.Conv1d(input_channels, internal_channels, 1, 1, 0)
        self.wn = WN(internal_channels, kernel_size, dilation_rate, num_resblock, speaker_encoding_channels)
        self.output_layer = nn.Conv1d(internal_channels, output_channels*2)

    def forward(self, x, speaker):
        x = self.input_layer(x)
        x = self.wn(x, speaker)
        x = self.output_layer(x)
        return x

