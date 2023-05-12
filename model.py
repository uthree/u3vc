import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as norm
import torch.nn.utils.weight_norm as wn
import torchaudio


class SpeakerEncoderResBlock(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 5, 1, 2)
        self.conv2 = nn.Conv1d(channels, channels, 5, 1, 2)
        self.act = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.conv2(self.act(self.conv1(x))) + x


class SpeakerEncoder(nn.Module):
    def __init__(self, dim_speaker=128):
        super().__init__()
        self.to_mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=22050,
                n_fft=512,
                n_mels=80,
                )
        self.layers = nn.Sequential(
                    nn.Conv1d(80, 32, 3, 1, 1),
                    SpeakerEncoderResBlock(32),
                    nn.Conv1d(32, 64, 4, 2, 1),
                    SpeakerEncoderResBlock(64),
                    nn.Conv1d(64, 128, 4, 2, 1),
                    SpeakerEncoderResBlock(128),
                    nn.Conv1d(128, 256, 4, 2, 1),
                    SpeakerEncoderResBlock(256),
                    nn.Conv1d(256, 512, 4, 2, 1)
                    )
        self.to_speaker = nn.Conv1d(512, dim_speaker * 2, 1, 1, 0)

    def forward(self, x):
        x = self.to_mel(x)
        x = self.layers(x).mean(dim=2, keepdim=True)
        mean, logvar = torch.chunk(self.to_speaker(x), 2, dim=1)
        return mean, logvar


class ContentEncoderResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.input_conv1 = wn(nn.Conv1d(channels, channels, 7, 1, dilation=1, padding='same'))
        self.input_conv2 = wn(nn.Conv1d(channels, channels, 7, 1, dilation=2, padding='same'))
        self.input_conv3 = wn(nn.Conv1d(channels, channels, 7, 1, dilation=3, padding='same'))
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        o1 = self.act(self.input_conv1(x))
        o2 = self.act(self.input_conv2(x))
        o3 = self.act(self.input_conv3(x))
        return o1 + o2 + o3 + x


class GeneratorResBlock(nn.Module):
    def __init__(self, channels, condition_channels=128):
        super().__init__()
        self.input_conv1 = wn(nn.Conv1d(channels, channels, 7, 1, dilation=1, padding='same'))
        self.condition_conv1 = wn(nn.Conv1d(condition_channels, channels, 1, 1, 0))
        self.input_conv2 = wn(nn.Conv1d(channels, channels, 7, 1, dilation=2, padding='same'))
        self.condition_conv2 = wn(nn.Conv1d(condition_channels, channels, 1, 1, 0))
        self.input_conv3 = wn(nn.Conv1d(channels, channels, 7, 1, dilation=3, padding='same'))
        self.condition_conv3 = wn(nn.Conv1d(condition_channels, channels, 1, 1, 0))
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x, c):
        o1 = self.act(self.input_conv1(x)) * self.condition_conv1(c)
        o2 = self.act(self.input_conv2(x)) * self.condition_conv2(c)
        o3 = self.act(self.input_conv3(x)) * self.condition_conv3(c)
        return o1 + o2 + o3 + x
        

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
    def __init__(self,
            channels,
            condition_channels=128,
            num_layers=2):
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
        self.initial_conv = nn.Conv1d(1, 32, 7, 1, 3)
        self.c1 = ContentEncoderResStack(32)
        self.d1 = nn.Conv1d(32, 64, 4, 2, 1)
        self.c2 = ContentEncoderResStack(64)
        self.d2 = nn.Conv1d(64, 128, 4, 2, 1)
        self.c3 = ContentEncoderResStack(128)
        self.d3 = nn.Conv1d(128, 256, 16, 8, 4)
        self.c4 = ContentEncoderResStack(256)
        self.d4 = nn.Conv1d(256, 256, 16, 8, 4)
        self.last_conv = nn.Conv1d(256, dim_content, 7, 1, 3)

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
        self.initial_conv = nn.Conv1d(dim_content, 256, 7, 1, 3)
        self.u1 = nn.ConvTranspose1d(256, 256, 16, 8, 4)
        self.c1 = GeneratorResStack(256)
        self.u2 = nn.ConvTranspose1d(256, 128, 16, 8, 4)
        self.c2 = GeneratorResStack(128)
        self.u3 = nn.ConvTranspose1d(128, 64, 4, 2, 1)
        self.c3 = GeneratorResStack(64)
        self.u4 = nn.ConvTranspose1d(64, 32, 4, 2, 1)
        self.c4 = GeneratorResStack(32)
        self.last_conv = nn.Sequential(
                nn.Conv1d(32, 32, 7, 1, 3),
                nn.LeakyReLU(0.1),
                nn.Conv1d(32, 1, 7, 1, 3))

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
        self.output_layers = nn.ModuleList([
                norm(nn.Conv1d(64, 1, 1, 1, 0)),
                norm(nn.Conv1d(64, 1, 1, 1, 0)),
                norm(nn.Conv1d(64, 1, 1, 1, 0)),
                norm(nn.Conv1d(64, 1, 1, 1, 0)),
                norm(nn.Conv1d(64, 1, 1, 1, 0)),
                norm(nn.Conv1d(64, 1, 1, 1, 0)),
            ])

    def logits(self, x):
        # [batch, len] -> [batch, 1, len]
        logits = []
        x = x.unsqueeze(1)
        x = self.initial_conv(x)
        for layer, to_out in zip(self.layers, self.output_layers):
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            logits.append(to_out(x))
        return logits

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
            out += d.logits(x)
        return out

    def feature_matching_loss(self, x, y):
        loss = 0
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
        self.input_layer = norm(
                nn.Conv2d(1, channels, (kernel_size, 1), (stride, 1), 0))
        self.layers = nn.Sequential()
        for i in range(num_stages):
            c = channels * (2 ** i)
            c_next = channels * (2 ** (i+1))
            if i == (num_stages - 1):
                self.layers.append(
                        norm(
                            nn.Conv2d(c, c, (kernel_size, 1), (stride, 1), groups=groups[i])))
            else:
                self.layers.append(
                        norm(
                            nn.Conv2d(c, c_next, (kernel_size, 1), (stride, 1), groups=groups[i])))
                self.layers.append(
                        nn.Dropout(dropout_rate))
                self.layers.append(
                        nn.LeakyReLU(0.2))
        c = channels * (2 ** (num_stages-1))
        self.final_conv = norm(
                nn.Conv2d(c, c, (5, 1), 1, 0)
                )
        self.final_relu = nn.LeakyReLU(0.1)
        self.output_layer = norm(
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
            loss += (x - y).abs().mean() / len(self.sub_discriminators)
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


class MelLoss(nn.Module):
    def __init__(self, sample_rate=22050, n_fft=1024, n_mels=80, normalized=True):
        super().__init__()
        self.to_mel = torchaudio.transforms.MelSpectrogram(sample_rate,
                                                           n_mels=n_mels,
                                                           n_fft=n_fft,
                                                           normalized=normalized,
                                                           hop_length=256)

    def forward(self, fake, real):
        self.to_mel = self.to_mel.to(real.device)
        with torch.no_grad():
            real_mel = self.to_mel(real)
        return ((self.to_mel(fake) - real_mel) ** 2).mean()


class SpectralLoss(nn.Module):
    def __init__(self, ws=[512, 1024, 2048]):
        super().__init__()
        self.mel_losses = nn.ModuleList([])
        for w in ws:
            self.mel_losses.append(MelLoss(n_fft=w))

    def forward(self, fake, real):
        loss = 0
        for mel_loss in self.mel_losses:
            loss += mel_loss(fake, real) / len(self.mel_losses)
        return loss
