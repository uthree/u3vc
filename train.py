import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

from tqdm import tqdm

from model import Convertor, Discriminator, SpectralLoss
from dataset import WaveFileDirectory

parser = argparse.ArgumentParser(description="run training")

parser.add_argument('dataset')
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-e', '--epoch', default=1000, type=int)
parser.add_argument('-b', '--batch-size', default=4, type=int)
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float)
parser.add_argument('-len', '--length', default=65536, type=int)
parser.add_argument('-m', '--max-data', default=-1, type=int)
parser.add_argument('-fp16', default=False, type=bool)

args= parser.parse_args()

def load_or_init_models(device=torch.device('cpu')):
    C = Convertor().to(device)
    D = Discriminator().to(device)
    if os.path.exists('./convertor.pt'):
        C.load_state_dict(torch.load('./convertor.pt', map_location=device))

    if os.path.exists('./dicriminator.pt'):
        D.load_state_dict(torch.load('./discriminator.pt', map_location=device))
    return C, D


def save_models(C, D):
    torch.save(C.state_dict(), './convertor.pt')
    torch.save(D.state_dict(), './discriminator.pt')


def write_preview(source_wave, file_path='./preview.wav'):
    source_wave = source_wave.detach().to(torch.float32).cpu()
    torchaudio.save(src=source_wave, sample_rate=22050, filepath=file_path)


device = torch.device(args.device)
C, D = load_or_init_models(device)

ds = WaveFileDirectory(
        [args.dataset],
        length=args.length,
        max_files=args.max_data
        )

dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size*2, shuffle=True)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

weight_rec = 10.0
weight_con = 10.0
weight_kl = 0.02
weight_spe = 1.0

OptC = optim.AdamW(C.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
OptD = optim.AdamW(D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

Es = C.speaker_encoder
Ec = C.content_encoder
G = C.generator

spectral_loss = SpectralLoss().to(device)

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, wave in enumerate(dl):
        wave_src, wave_tgt = torch.chunk(wave.to(device), 2, dim=0)

        # Train Convertor.
        OptC.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):

            # Encode Speaker
            mean_src, logvar_src = Es(wave_src)
            c_src = mean_src + torch.exp(logvar_src) * torch.randn(*mean_src.shape, device=device)
            mean_tgt, logvar_tgt = Es(wave_tgt)
            c_tgt = mean_tgt = torch.exp(logvar_src) * torch.randn(*mean_tgt.shape, device=device)

            # Reconstruction Loss
            z_src = Ec(wave_src)
            rec_out = G(z_src, c_src)

            loss_fm = D.feature_matching_loss(rec_out, wave_src)
            loss_spe = spectral_loss(rec_out, wave_src)
            loss_rec = loss_fm + weight_spe * loss_spe

            # Adversarial Loss
            convert_out = G(z_src, c_tgt)
            loss_adv = 0
            for logit in D.logits(convert_out):
                loss_adv += (logit ** 2).mean()

            # Content Preservation Loss
            loss_con = ((Ec(wave_src) - Ec(convert_out)) ** 2).mean()

            # KL Loss
            loss_kl = (-1 - logvar_src + torch.exp(logvar_src) + mean_src ** 2).mean()

            # Final Loss
            loss_convertor = loss_adv + weight_rec * loss_rec + weight_con * loss_con + weight_kl * loss_kl

        scaler.scale(loss_convertor).backward()
        torch.nn.utils.clip_grad_norm_(C.parameters(), 1.0)
        scaler.step(OptC)
        
        # Train Discriminator
        convert_out = convert_out.detach()
        OptD.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            loss_d = 0
            for logit in D.logits(convert_out):
                loss_d += ((logit - 1) ** 2).mean()
            for logit in D.logits(wave_src):
                loss_d += ((logit) ** 2).mean()

        scaler.scale(loss_d).backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
        scaler.step(OptD)

        scaler.update()

        bar.set_description(f"G: {loss_convertor.item():.4f}, D: {loss_d.item():.4f}")
        tqdm.write(f"Adv.: {loss_adv.item():.4f}, Spe.: {loss_spe.item():.4f}, F.M.: {loss_fm.item():.4f}, Con.: {loss_con.item():.4f}, K.L.: {loss_kl.item():.4f}")

        N = wave.shape[0]
        bar.update(N)
        if batch % 100 == 0:
            print("Saving Models...")
            save_models(C, D)
            print("Complete!")
            print("Writing Previews")
            write_preview(wave_tgt[0].unsqueeze(0), './target.wav')
            write_preview(wave_src[0].unsqueeze(0), './source.wav')
            write_preview(convert_out[0].unsqueeze(0), './output.wav')
            print("Complete!")
