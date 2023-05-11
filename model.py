import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.conv1 = norm(nn.Conv1d(channels, channels, 5, 1, 2))
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = norm(nn.Conv1d(channels, channels, 5, 1, 2))

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x))) + x


class GeneratorResBlock(nn.Module):
    def __init__(self, channels, condition_channels=256):
        super().__init__()
        self.conv1 = norm(nn.Conv1d(channels, channels, 5, 1, 2))
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = norm(nn.Conv1d(channels, channels, 5, 1, 2))
        self.condition_conv = norm(nn.Conv1d(condition_channels, channels, 1, 1, 0))

    def forward(self, x, c):
        res = x
        x = x * torch.sigmoid(self.condition_conv(c))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + res


class ContentEncoderResStack(nn.Module):
    def __init__(self, channels, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(ContentEncoderResBlock(channels))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class GeneratorResStack(nn.Module):
    def __init__(self, channels, condition_channels=256, num_layers=4):
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
        self.u1 = norm(nn.ConvTranspose1d(256, 256, 16, 8, 4))
        self.c1 = GeneratorResStack(256)
        self.u2 = norm(nn.ConvTranspose1d(256, 128, 16, 8, 4))
        self.c2 = GeneratorResStack(128)
        self.u3 = norm(nn.ConvTranspose1d(128, 64, 4, 2, 1))
        self.c3 = GeneratorResStack(64)
        self.u4 = norm(nn.ConvTranspose1d(64, 32, 4, 2, 1))
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
        x = x.squeeze(1)
        # [batch, 1, len] -> [batch, len]
        x = torch.tanh(x)
        return x


class SubDiscriminator(nn.Module):
    def __init__(self, pool=1):
        super().__init__()
        self.pool = nn.AvgPool1d(pool)
        self.initial_conv = norm(nn.Conv1d(1, 32, 7, 1, 3))
        self.layers = nn.ModuleList([
            norm(nn.Conv1d(32, 64, 41, 2)),
            norm(nn.Conv1d(64, 64, 41, 2)),
            norm(nn.Conv1d(64, 64, 41, 4)),
            norm(nn.Conv1d(64, 64, 41, 4)),
            norm(nn.Conv1d(64, 64, 41, 4)),
            norm(nn.Conv1d(64, 64, 41, 4)),
            ])
        self.output_layer = norm(nn.Conv1d(64, 1, 7, 1, 3))

    def forward(self, x):
        # [batch, len] -> [batch, 1, len]
        x = x.unsqueeze(1)
        x = self.initial_conv(x)
        for layer in self.layers:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
        x = self.output_layer(x)
        return x

    def feature_matching_loss(self, x, y):
        x = x.unsqueeze(1)
        x = self.initial_conv(x)
        with torch.no_grad():
            y = y.unsqueeze(1)
            y = self.initial_conv(y)
        loss = 0
        for layer in self.layers:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            with torch.no_grad():
                y = layer(y)
                y = F.leaky_relu(y, 0.1)
            loss += (x - y).abs().mean()
        return loss


class Discriminator(nn.Module):
    def __init__(self, pools=[1, 2, 4]):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([])
        for p in pools:
            self.sub_discriminators.append(SubDiscriminator(p))

    def logits(self, x):
        out = []
        for d in self.sub_discriminators:
            out.append(d(x))
        return out

    def feature_matching_loss(self, x, y):
        loss = (x - y).abs().mean()
        for d in self.sub_discriminators:
            loss += d.feature_matching_loss(x, y)
        return loss


class Convertor(nn.Module):
    def __init__(self):
        super().__init__()
        self.speaker_encoder = SpeakerEncoder()
        self.content_encoder = ContentEncoder()
        self.generator = Generator()

    def encode_speaker(self, wave):
        mean, _ = self.speaker_encoder(wave)
        return mean

    def convert(self, wave, target_speaker):
        z = self.content_encoder(wave)
        y = self.generator(z, target_speaker)
        return y


class SpectralLoss(nn.Module):
    def __init__(self, ws=[2048, 1024, 512]):
        super().__init__()
        self.to_mels = nn.ModuleList([])
        for w in ws:
            to_mel = torchaudio.transforms.MelSpectrogram(
                    n_fft = w,
                    n_mels = 80,
                    sample_rate = 22050,
                    )
            self.to_mels.append(to_mel)

    def forward(self, x, y, eps=1e-4):
        loss = 0
        for to_mel in self.to_mels:
            x_mel = torch.tanh(to_mel(x))
            y_mel = torch.tanh(to_mel(y))
            loss += ((x_mel - y_mel) ** 2).mean()
        if torch.any(loss.isinf()) or torch.any(loss.isnan()):
            loss = torch.tensor(0, device=x.device)
        return loss
