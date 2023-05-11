import argparse
import glob
import os

import torch
import torchaudio
from  torchaudio.functional import resample as resample

from tqdm import tqdm

from model import Convertor

parser = argparse.ArgumentParser(description="inference")

parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-i', '--inputs', default='./inputs')
parser.add_argument('-t', '--target-speaker', default='./speaker.wav')

args = parser.parse_args()

device = torch.device(args.device)

convertor = Convertor().to(device)

if os.path.exists('./convertor.pt'):
    print("Loading Model...")
    convertor.load_state_dict(torch.load('./convertor.pt', map_location=device))

target_wav, sr = torchaudio.load(args.target_speaker)
target_wav = resample(target_wav, sr, 22050)
target_wav = target_wav.to(device)

print("Encoding target speaker...")
target_speaker = convertor.encode_speaker(target_wav)

if not os.path.exists("./outputs/"):
    os.mkdir("outputs")

paths = glob.glob(os.path.join(args.inputs, "*.wav"))
for i, path in enumerate(paths):
    wf, sr = torchaudio.load(path)
    print(f"converting {path}")
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=args.fp16):
            wf = resample(wf, sr, 22050)
            wf = wf.to(device)
            wf = convertor.convert(wf, target_speaker)
            wf = resample(wf, 22050, sr)
    wf = wf.to('cpu').detach()
    torchaudio.save(src=wf, sample_rate=sr, filepath=f"./outputs/out_{i}.wav")
    
