import argparse
import os

import torch
import torchaudio
from torchaudio.functional import resample as resample

from model import VoiceConvertor
from dataset import WaveFileDirectory
from module.hubert import load_hubert, interpolate_hubert_output

parser = argparse.ArgumentParser(description="Inference")

parser.add_argument('-d', '--device', default='cpu',
                    help="Device setting. Set this option to cuda if you need to use ROCm.")
parser.add_argument('-i', '--input', default='./inputs',
                    help="Input directory")
parser.add_argument('-o', '--output', default='./outputs',
                    help="Output directory")
parser.add_argument('-t', '--target', default='./target.wav')


args = parser.parse_args()

device = torch.device(args.device)

model = VoiceConvertor().to(device)
model.load_state_dict(torch.load('./model.pt', map_location=device))

to_spectrogram = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256).to(device)
to_melspectrogram = torchaudio.transforms.MelSpectrogram(n_fft=1024, n_mels=80, normalized=True).to(device)

Es = model.convertor.speaker_encoder
Dec = model.convertor.decoder
Flow = model.convertor.flow
Ep = model.convertor.posterior_encoder


if not os.path.exists(args.output):
    os.mkdir(args.output)

wf, sr = torchaudio.load(args.target)
wf = resample(wf, sr, 16000)
wf = wf.to(device)
tgt_spk = Es(to_melspectrogram(wf))

for i, fname in enumerate(os.listdir(args.input)):
    print(f"Inferencing {fname}")
    with torch.no_grad():
        wf, sr = torchaudio.load(os.path.join(args.input, fname))
        wf = resample(wf, sr, 16000)
        wf = wf.to(device)

        src_spk = Es(to_melspectrogram(wf))
        spec = to_spectrogram(wf)[:, :, 1:]
        mean_phi, logvar_phi = Ep(spec, src_spk)
        z = mean_phi + torch.exp(logvar_phi) * torch.randn_like(logvar_phi)
        z = Flow(z, src_spk, reverse=True)
        z = Flow(z, tgt_spk)
        wf = Dec(z, tgt_spk)

        wf = resample(wf, 16000, sr)
        wf = wf.to(torch.device('cpu'))
        out_path = os.path.join(args.output, f"output_{fname}_{i}.wav")
        torchaudio.save(out_path, src=wf, sample_rate=sr)

