import torch
import torch.nn as nn
import torchaudio

from module.hubert import load_hubert, interpolate_hubert_output

def match_features(source, reference, k=4):
    with torch.no_grad():
        # source: [N, 768, Length], reference: [N, 768, Length]
        source = source.transpose(1, 2)
        reference = reference.transpose(1, 2)
        source_norm = torch.norm(source, dim=2, keepdim=True)
        reference_norm = torch.norm(reference, dim=2, keepdim=True)
        cos_sims = torch.bmm((source / source_norm), (reference / reference_norm).transpose(1, 2))
        best = torch.topk(cos_sims, k, dim=2)
        result = torch.stack([reference[n][best.indices[n]] for n in range(source.shape[0])], dim=0).mean(dim=2)
        result = result.transpose(1, 2)
    return result


class ContentExtracctor(nn.Module):
    def __init__(self, lut, hubert_dim=768, internal_channels=96):
        super().__init__()
        self.lut = nn.Parameter(lut)
        self.proj = nn.Conv1d(hubert_dim, internal_channels, 1, 1, 0)

    def forward(self, x):
        features = match_features(x, self.lut).detach()
        return self.proj(features)


def from_intermediate_audio_file(path="intermediate.wav", device=torch.device('cpu')):
    hubert = load_hubert(device)
    wave, sr = torchaudio.load(path)
    wave = torchaudio.functional.resample(wave, sr, 16000)
    hubert_output = interpolate_hubert_output(hubert(wave), wave.shape[1])
    ce = ContentExtracctor(hubert_output)
    ce = ce.to(device)
    return ce
