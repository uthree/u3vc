import torch
import torch.nn as nn


class WNBlock(nn.Module):
    def __init__(self,
            hidden_channels=96,
            kernel_size=5,
            dilation_rate=1,
            speaker_encoding_channels=128,
            with_speaker=True
            ):
        super().__init__()
        self.with_speaker = with_speaker
        if with_speaker:
            self.speaker_conv = nn.Conv1d(speaker_encoding_channels, hidden_channels*2, 1, 1, 0)
        self.input_conv = nn.Conv1d(hidden_channels, hidden_channels*2, kernel_size, 1, dilation=dilation_rate, padding='same')
        self.output_conv = nn.Conv1d(hidden_channels, hidden_channels, 1, 1, 0)

    def forward(self, x, speaker=None):
        res = x
        if self.with_speaker:
            x = self.input_conv(x) + self.speaker_conv(speaker)
        else:
            x = self.input_conv(x)
        x_t, x_s = torch.chunk(x, 2, dim=1)
        x_t = torch.tanh(x_t)
        x_s = torch.sigmoid(x_s)
        x = x_t * x_s
        return x + res, x


class WN(nn.Module):
    def __init__(self,
            hidden_channels=96,
            kernel_size=5,
            dilation_rate=1,
            num_blocks=4,
            speaker_encoding_channels=128,
            with_speaker=True):

        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(WNBlock(hidden_channels, kernel_size, dilation_rate, speaker_encoding_channels, with_speaker))

    def forward(self, x, speaker=None):
        outputs = 0
        for block in self.blocks:
            x, out = block(x, speaker)
            outputs = outputs + out
        return outputs
