import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

from tqdm import tqdm

from model import VoiceConvertor
from dataset import WaveFileDirectory
from module.hubert import load_hubert, interpolate_hubert_output

parser = argparse.ArgumentParser(description="run training")

parser.add_argument('dataset')
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-e', '--epoch', default=1000, type=int)
parser.add_argument('-b', '--batch-size', default=1, type=int)
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float)
parser.add_argument('-len', '--length', default=32768, type=int)
parser.add_argument('-m', '--max-data', default=-1, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-gacc', '--gradient-accumulation', default=1, type=int)

args = parser.parse_args()

def load_or_init_models(device=torch.device('cpu')):
    model = VoiceConvertor().to(device)
    if os.path.exists('./model.pt'):
        model.load_state_dict(torch.load('./model.pt', map_location=device))
    return model


def save_models(model):
    print("Saving Models...")
    torch.save(model.state_dict(), './model.pt')
    print("complete!")


def write_preview(source_wave, file_path='./preview.wav'):
    source_wave = source_wave.detach().to(torch.float32).cpu()
    torchaudio.save(src=source_wave, sample_rate=16000, filepath=file_path)


def cut_center_wave(wave):
    length = wave.shape[1]
    s = length//2 - length//4
    e = length//2 + length//4
    return wave[:, s:e]


def cut_center_spec(spec):
    length = spec.shape[2]
    s = length//2 - length//4
    e = length//2 + length//4
    return spec[:, :, s:e]


device = torch.device(args.device)
model = load_or_init_models(device)

ds = WaveFileDirectory(
        [args.dataset],
        length=args.length,
        max_files=args.max_data
        )

dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

OptC = optim.Adam(model.convertor.parameters(), lr=args.learning_rate)
OptD = optim.Adam(model.discriminator.parameters(), lr=args.learning_rate)

Dis = model.discriminator
Eb = model.convertor.bottleneck_extractor
Es = model.convertor.speaker_encoder
Dec = model.convertor.decoder
Flow = model.convertor.flow
Ep = model.convertor.posterior_encoder

hubert = load_hubert(device)

to_spectrogram = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256).to(device)
to_melspectrogram = torchaudio.transforms.MelSpectrogram(n_fft=1024, n_mels=80).to(device)

grad_acc = args.gradient_accumulation

weight_l1 = 5.0
weight_kl = 1.0
weight_fm = 2.0
weight_mel = 45.0

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, wave in enumerate(dl):
        wave = wave.to(device)
        spec = to_spectrogram(wave)[:, :, 1:]
        melspec = to_melspectrogram(wave)[:, :, 1:]
        
        # Train Convertor.
        with torch.cuda.amp.autocast(enabled=args.fp16):
            spk = Es(melspec)
            mean_phi, logvar_phi = Ep(spec, spk)
            z_phi = mean_phi + torch.exp(logvar_phi) * torch.randn_like(logvar_phi)
            wave_out = Dec(z_phi, spk)
            mel_out = to_melspectrogram(wave_out)[:, :, 1:]
            loss_mel = (cut_center_spec(melspec) - cut_center_spec(mel_out)).abs().mean()
            loss_fm = Dis.feat_loss(cut_center_wave(wave_out), cut_center_wave(wave))
            logits = Dis.logits(wave_out)
            loss_adv = 0
            for logit in logits:
                loss_adv += (logit ** 2).mean()
            ssl = interpolate_hubert_output(hubert(wave), wave.shape[1])
            mean_theta, logvar_theta = Eb(ssl)
            z_theta = mean_theta + torch.exp(logvar_theta) * torch.randn_like(logvar_theta)
            flow_out = Flow(z_phi, spk, reverse=True)
            loss_l1 = (z_theta - flow_out).abs().mean()
            loss_kl = (-1 -logvar_phi + torch.exp(logvar_phi) + mean_phi ** 2).mean() +\
                    (-1 -logvar_theta + torch.exp(logvar_theta) + mean_theta ** 2).mean()
            # nan reduction
            if torch.any(torch.isnan(loss_mel)):
                loss_mel = torch.tensor(0).to(device)
            loss_c = loss_adv + loss_fm * weight_fm + loss_mel * weight_mel + loss_l1 * weight_l1 + loss_kl * weight_kl

        scaler.scale(loss_c).backward()

        if batch % grad_acc == 0:
            scaler.step(OptC)
            OptC.zero_grad()


        # Train Discriminator.
        OptD.zero_grad()
        wave_out = wave_out.detach() # Fake wave
        with torch.cuda.amp.autocast(enabled=args.fp16):
            loss_d = 0
            logits = Dis.logits(wave_out)
            for logit in logits:
                loss_d += ((logit - 1) ** 2).mean()
            logits = Dis.logits(wave)
            for logit in logits:
                loss_d += ((logit) ** 2).mean()
        scaler.scale(loss_d).backward()
        scaler.step(OptD)

        scaler.update()
        
        tqdm.write(f"D: {loss_d.item():.4f}, Adv.: {loss_adv.item():.4f}, F.M.: {loss_fm.item():.4f}, Mel.: {loss_mel.item():.4f}, L1: {loss_l1.item():.4f}, K.L.: {loss_kl.item():.4f}")

        N = wave.shape[0]
        bar.update(N)

        if batch % 100 == 0:
            save_models(model)

print("Training Complete!")
save_models(model)
