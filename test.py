import torch
import torchaudio


from module.posterior_encoder import PosteriorEncoder
from module.decoder import Decoder
from module.discriminator import Discriminator

posterior_encoder = PosteriorEncoder()
decoder = Decoder()
discriminator = Discriminator()

wave = torch.randn(1, 65536)
spk = torch.randn(1, 256, 1)

to_spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=1024,
        hop_length=256
        )

spec = to_spectrogram(wave)[:, :, 1:]
mu, sigma = posterior_encoder(spec, spk)
z = mu + torch.exp(sigma) * torch.randn_like(sigma)
out = decoder(z, spk)
print(out.shape)
