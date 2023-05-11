import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as norm
import torchaudio


class SpeakerEncoder(nn.Module):
    def __init__(self, dim_speaker=256):
        super().__init__()
        self.to_mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=22050,
                n_fft=512,
                n_mels=80,
                )
        self.layers = nn.Sequential(
                    norm(nn.Conv1d(80, 32, 3, 1, 1)),
                    nn.LeakyReLU(0.1),
                    norm(nn.Conv1d(32, 64, 3, 1, 1)),
                    nn.LeakyReLU(0.1),
                    nn.AvgPool1d(2),
                    norm(nn.Conv1d(64, 128, 3, 1, 1)),
                    nn.LeakyReLU(0.1),
                    nn.AvgPool1d(2),
                    norm(nn.Conv1d(128, 256, 3, 1, 1)),
                    nn.LeakyReLU(0.1),
                    nn.AvgPool1d(2),
                    norm(nn.Conv1d(256, 512, 3, 1, 1)),
                    nn.LeakyReLU(0.1),
                    )
        self.to_speaker = norm(nn.Conv1d(512, dim_speaker * 2, 1, 1, 0))

    def forward(self, x):
        x = self.to_mel(x)
        x = self.layers(x).mean(dim=2, keepdim=True)
        mean, logvar = torch.chunk(self.to_speaker(x), 2, dim=1)
        return mean, logvar


class ContentEncoderResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = norm(nn.Conv1d(channels, channesl, 5, 1, 2))
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = norm(nn.Conv1d(channels, channesl, 5, 1, 2))

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x))) + x


class GeneratorResBlock(nn.Module):
    def __init__(self, channels, condition_channels=256):
        super().__init__()
        self.conv1 = norm(nn.Conv1d(channels, channesl, 5, 1, 2))
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = norm(nn.Conv1d(channels, channesl, 5, 1, 2))
        self.condition_conv = norm(nn.Conv1d(condition_channels, channels))

    def forward(self, x, c):
        res = x
        x = x * self.condition_conv(c)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + res


class ContentEncoderResStack(nn.Module):
    def __init__(self, channels, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(ContentEncoderResBlock(channels))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class GeneratorResStack(nn.Module):
    def __init__(self, channels, condition_channels=256, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(GeneratorResBlock(channels, condition_channels))

    def forward(self, x, c):
        for layer in self.layers:
            x = layer(x, c)
        return x


class ContentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_conv = norm(nn.Conv1d(1, 32, 7, 1, 3))
        self.c1 = ContentEncoderResStack(32)
        self.d1 = norm(nn.Conv1d(32, 64, 4, 2, 1))
        self.c2 = ContentEncoderResStack(64)
        self.d2 = norm(nn.Conv1d(64, 128, 4, 2, 1))
        self.c3 = ContentEncoderResStack(128)
        self.d3 = norm(nn.Conv1d(128, 256, 16, 8, 4))
        self.c4 = ContentEncoderResStack(256)
        self.d4 = norm(nn.Conv1d(256, 256, 16, 8, 4))
        self.last_conv = norm(nn.Conv1d(256, 256, 7, 1, 3))

    def forward(self, x):
        # x: [batch, len]
        x = x.unsqueeze(1)
        # x: [batch, 1, len]
        x = self.initial_conv(x)
        x = self.c1(x)
        x = self.d1(x)
        x = self.c2(x)
        x = self.d2(x)
        x = self.c3(x)
        x = self.d3(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.last_conv(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_conv = norm(nn.Conv1d(256, 256, 7, 1, 3))
        self.u1 = norm(nn.Conv1d(256, 256, 16, 8, 4))
        self.c1 = GeneratorResStack(256)
        self.u2 = norm(nn.Conv1d(256, 128, 16, 8, 4))
        self.c2 = GeneratorResStack(128)
        self.u3 = norm(nn.Conv1d(128, 64, 4, 2, 1))
        self.c3 = GeneratorResStack(64)
        self.u4 = norm(nn.Conv1d(64, 32, 4, 2, 1))
        self.c4 = GeneratorResStack(32)
        self.last_conv = norm(nn.Conv1d(32, 1, 7, 1, 3))

    def forward(self, x, c):
        x = self.initial_conv(x)
        x = self.u1(x)
        x = self.c1(x, c)
        x = self.u2(x)
        x = self.c2(x, c)
        x = self.u3(x)
        x = self.c3(x, c)
        x = self.u4(x)
        x = self.c4(x, c)
        x = self.last_conv(x)
        return x
