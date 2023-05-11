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


class MRFResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.c1 = norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                            padding='same', dilation=dilation))
        self.c2 = norm(nn.Conv1d(channels, channels, kernel_size, 1,
                            padding='same', dilation=dilation))
        self.act1 = nn.LeakyReLU(0.1)
        self.act2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.c1(self.act1(x)) + x
        x = self.c2(self.act2(x)) + x
        return x


class MRF(nn.Module):
    def __init__(self, channels, kernel_sizes=[3, 5, 7], dilations=[1, 2, 3]):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for d, k in zip(dilations, kernel_sizes):
            self.blocks.append(MRFResBlock(channels, k, d))

    def forward(self, x):
        output = 0
        for block in self.blocks:
            output = output + block(x)
        return output


class GeneratorResBlock(nn.Module):
    def __init__(self, channels, condition_channels=256):
        super().__init__()
        self.condition_conv = norm(nn.Conv1d(condition_channels, channels, 1, 1, 0))
        self.mrf = MRF(channels)

    def forward(self, x, c):
        res = x
        x = x * torch.sigmoid(self.condition_conv(c))
        x = self.mrf(x)
        return x + res


class ContentEncoderResStack(nn.Module):
    def __init__(self, channels, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(ContentEncoderResBlock(channels))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class GeneratorResStack(nn.Module):
    def __init__(self, channels, condition_channels=256, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(GeneratorResBlock(channels, condition_channels))

    def forward(self, x, c):
        for layer in self.layers:
            x = layer(x, c)
        return x


class ContentEncoder(nn.Module):
    def __init__(self, dim_content=4):
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
        self.last_conv = norm(nn.Conv1d(256, dim_content, 7, 1, 3))

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
    def __init__(self, dim_content=4):
        super().__init__()
        self.initial_conv = norm(nn.Conv1d(dim_content, 256, 7, 1, 3))
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


class SubDiscriminatorS(nn.Module):
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


class DiscriminatorS(nn.Module):
    def __init__(self, pools=[1, 2, 4]):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([])
        for p in pools:
            self.sub_discriminators.append(SubDiscriminatorS(p))

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


class SubDiscriminatorP(nn.Module):
    def __init__(self,
                 channels=32,
                 period=2,
                 kernel_size=5,
                 stride=3,
                 num_stages=4,
                 dropout_rate=0.2,
                 groups = []
                 ):
        super().__init__()
        self.input_layer = nn.utils.spectral_norm(
                nn.Conv2d(1, channels, (kernel_size, 1), (stride, 1), 0))
        self.layers = nn.Sequential()
        for i in range(num_stages):
            c = channels * (2 ** i)
            c_next = channels * (2 ** (i+1))
            if i == (num_stages - 1):
                self.layers.append(
                        nn.utils.spectral_norm(
                            nn.Conv2d(c, c, (kernel_size, 1), (stride, 1), groups=groups[i])))
            else:
                self.layers.append(
                        nn.utils.spectral_norm(
                            nn.Conv2d(c, c_next, (kernel_size, 1), (stride, 1), groups=groups[i])))
                self.layers.append(
                        nn.Dropout(dropout_rate))
                self.layers.append(
                        nn.LeakyReLU(0.2))
        c = channels * (2 ** (num_stages-1))
        self.final_conv = nn.utils.spectral_norm(
                nn.Conv2d(c, c, (5, 1), 1, 0)
                )
        self.final_relu = nn.LeakyReLU(0.1)
        self.output_layer = nn.utils.spectral_norm(
                nn.Conv2d(c, 1, (3, 1), 1, 0))
        self.period = period

    def logit(self, x):
        # padding
        if x.shape[1] % self.period != 0:
            pad_len = self.period - (x.shape[1] % self.period)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)

        x = x.view(x.shape[0], self.period, -1)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.input_layer(x)
        x = self.layers(x)
        x = self.final_conv(x)
        x = self.final_relu(x)
        x = self.output_layer(x)
        return x

    def feats(self, x):
        # padding
        if x.shape[1] % self.period != 0:
            pad_len = self.period - (x.shape[1] % self.period)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)

        x = x.view(x.shape[0], self.period, -1)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.input_layer(x)
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats


class DiscriminatorP(nn.Module):
    def __init__(self,
                 periods=[2, 3, 5, 7, 11],
                 groups=[1, 1, 1, 1],
                 channels=64,
                 kernel_size=5,
                 stride=3,
                 num_stages=4,
                 ):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([])

        for p in periods:
            self.sub_discriminators.append(
                    SubDiscriminatorP(channels,
                                          p,
                                          kernel_size,
                                          stride,
                                          num_stages,
                                          groups=groups))

    def logits(self, x):
        logits = []
        for sd in self.sub_discriminators:
            logits.append(sd.logit(x))
        return logits

    def feature_matching_loss(self, x, y):
        feats_x = []
        feats_y = []
        loss = 0
        for sd in self.sub_discriminators:
            feats_x = feats_x + sd.feats(x)
        for sd in self.sub_discriminators:
            feats_y = feats_y + sd.feats(y)
        for x, y in zip(feats_x, feats_y):
            loss += (x - y).abs().mean()
        return loss


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_p = DiscriminatorP()
        self.d_s = DiscriminatorS()

    def logits(self, x):
        return self.d_p.logits(x) + self.d_s.logits(x)

    def feature_matching_loss(self, x, y):
        return self.d_p.feature_matching_loss(x, y) + self.d_s.feature_matching_loss(x, y)


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
            x_mel = torch.log10(to_mel(x).abs() + eps)
            y_mel = torch.log10(to_mel(y).abs() + eps)
            l = ((x_mel - y_mel) ** 2).mean() / len(self.to_mels)
            if torch.any(l.isinf()) or torch.any(l.isnan()):
                l = torch.tensor(0., device=x.device)
            loss += l
        return loss
