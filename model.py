import torch
import torch.nn as nn


class ContentEncoderResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.input_conv = nn.Conv1d(channels, channels, 1, 1, 0)
        self.res_conv = nn.Conv1d(channels, channels, 1, 1, 0)
        self.output_conv = nn.Conv1d(channels, channels, 1, 1, 0)

    def forward(self, x):
        res = self.res_conv(x)
        x = self.input_conv(x)
        x = torch.sigmoid(x) * torch.tanh(x)
        x = self.output_conv(x)
        return x + res


class GeneratorResBlock(nn.Module):
    def __init__(self, channels, condition_channels):
        super().__init__()
