import torch
import torch.nn as nn


LRELU_SLOPE = 0.1


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 5, 1, 2)
        self.act = nn.LeakyReLU(LRELU_SLOPE)
        self.conv2 = nn.Conv1d(channels, channels, 5, 1, 2)
    
    def forward(self, x):
        return self.conv2(self.act(self.conv1(x))) + x


class SpeakerEncoder(nn.Module):
    def __init__(self,
            input_channels=80,
            internal_channels=128,
            num_resblock=7,
            speaker_encoding_channels=128):
        super().__init__()
        self.input_conv = nn.Conv1d(input_channels, internal_channels, 1, 1, 0)
        self.mid_layers = nn.Sequential(*[ResBlock(internal_channels) for _ in range(num_resblock)])
        self.output_conv = nn.Conv1d(internal_channels, speaker_encoding_channels, 1, 1, 0)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.mid_layers(x)
        x = x.mean(dim=2, keepdim=True)
        x = self.output_conv(x)
        return x
