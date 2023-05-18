import torch
import torch.nn as nn


class WNBlock(nn.Module):
    def __init__(self,
            hidden_channels=192,
            kernel_size=5,
            dilation_rate=1,
            speaker_encoding_channels=256,
            ):
        super().__init__()
        self.speaker_conv = nn.Conv1d(speaker_encoding_channels, hidden_channels*2, 1, 1, 0)
        self.input_conv = nn.Conv1d(hidden_channels, hidden_channels*2, kernel_size, 1, dilation=dilation_rate, padding='same')
        self.output_conv = nn.Conv1d(hidden_channels, hidden_channels, 1, 1, 0)

    def forward(self, x, speaker):
        res = x
        x = self.input_conv(x) + self.speaker_conv(speaker)
        x_t, x_s = torch.chunk(x, 2, dim=1)
        x_t = torch.tanh(x_t)
        x_s = torch.sigmoid(x_s)
        x = x_t * x_s
        return x + res, x


class WN(nn.Module):
    def __init__(self,
            hidden_channels=192,
            kernel_size=5,
            dilation_rate=1,
            num_blocks=4,
            speaker_encoding_channels=256):

        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(WNBlock(hidden_channels, kernel_size, dilation_rate, speaker_encoding_channels))

    def forward(self, x, speaker):
        outputs = 0
        for block in self.blocks:
            x, out = block(x, speaker)
            outputs = outputs + out
        return outputs
