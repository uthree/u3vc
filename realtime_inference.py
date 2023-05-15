import argparse
import pyaudio
import os

import torch
import torchaudio
from  torchaudio.functional import resample as resample

from tqdm import tqdm
import numpy as np

from model import Convertor

parser = argparse.ArgumentParser(description="inference")

parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-t', '--target-speaker', default='./speaker.wav')
parser.add_argument('-i', '--input', default=0, type=int)
parser.add_argument('-o', '--output', default=0, type=int)
parser.add_argument('-l', '--loopback', default=-1, type=int)
parser.add_argument('-ig', '--input-gain', default=1.0, type=float)
parser.add_argument('-g', '--gain', default=1.0, type=float)
parser.add_argument('-c', '--chunk-size', default=4096, type=int)
parser.add_argument('-b', '--buffer-size', default=4, type=int)

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

audio = pyaudio.PyAudio()

stream_input = audio.open(
        format=pyaudio.paInt16,
        rate=44100,
        channels=1,
        input_device_index=args.input,
        input=True)
stream_output = audio.open(
        format=pyaudio.paInt16,
        rate=44100,
        channels=1,
        output_device_index=args.output,
        output=True)
stream_loopback = audio.open(
        format=pyaudio.paInt16,
        rate=44100,
        channels=1,
        output_device_index=args.loopback,
        output=True) if args.loopback != -1 else None


buffer = []
chunk = args.chunk_size
buffer_size = args.buffer_size

print("Conveting Voice...")
while True:
    wave = stream_input.read(chunk)
    wave = np.frombuffer(wave, dtype=np.int16)
    buffer.append(wave)

    if len(buffer) > buffer_size:
        del buffer[0]
    else:
        continue

    wave = np.concatenate(buffer, 0)
    wave = wave.astype(np.float32) / 32768 # Convert -1 to 1
    wave = torch.from_numpy(wave).to(device).unsqueeze(0) * args.input_gain
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=args.fp16):
            wave = resample(wave, 44100, 22050)
            wave = convertor.convert(wave, target_speaker)
            wave = resample(wave, 22050, 44100)
            
            wave = wave[0]
    wave = wave.cpu().numpy()
    wave = wave * 32768 * args.gain
    wave = wave.astype(np.int16)

    b = (chunk * buffer_size) // 2 - (chunk // 2)
    e = (chunk * buffer_size) - b
    wave = wave[b:e].tobytes()

    stream_output.write(wave)
    
    if stream_loopback is not None:
        stream_loopback.write(wave)
