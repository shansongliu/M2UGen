import torch.cuda
import sys

sys.path.append('../../M2UGen')

import tempfile
from PIL import Image
import scipy
import argparse

from llama.m2ugen import M2UGen
import llama
import numpy as np
import os
import torch
import torchaudio
import torchvision.transforms as transforms
import av
import subprocess
import librosa
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="./ckpts/checkpoint.pth", type=str,
    help="Name of or path to M2UGen pretrained checkpoint",
)
parser.add_argument(
    "--llama_type", default="7B", type=str,
    help="Type of llama original weight",
)
parser.add_argument(
    "--llama_dir", default="/path/to/llama", type=str,
    help="Path to LLaMA pretrained checkpoint",
)
parser.add_argument(
    "--mert_path", default="m-a-p/MERT-v1-330M", type=str,
    help="Path to MERT pretrained checkpoint",
)
parser.add_argument(
    "--vit_path", default="m-a-p/MERT-v1-330M", type=str,
    help="Path to ViT pretrained checkpoint",
)
parser.add_argument(
    "--vivit_path", default="m-a-p/MERT-v1-330M", type=str,
    help="Path to ViViT pretrained checkpoint",
)
parser.add_argument(
    "--knn_dir", default="./ckpts", type=str,
    help="Path to directory with KNN Index",
)
parser.add_argument(
    '--music_decoder', default="musicgen", type=str,
    help='Decoder to use musicgen/audioldm2')

parser.add_argument(
    '--music_decoder_path', default="facebook/musicgen-medium", type=str,
    help='Path to decoder to use musicgen/audioldm2')

args = parser.parse_args()

llama_type = args.llama_type
llama_ckpt_dir = os.path.join(args.llama_dir, llama_type)
llama_tokenzier_path = args.llama_dir
model = M2UGen(llama_ckpt_dir, llama_tokenzier_path, args, knn=False, stage=3, load_llama=False)

print("Loading Model Checkpoint")
checkpoint = torch.load(args.model, map_location='cpu')

new_ckpt = {}
for key, value in checkpoint['model'].items():
    key = key.replace("module.", "")
    new_ckpt[key] = value

load_result = model.load_state_dict(new_ckpt, strict=True)
assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"
model.eval()
model.to("cuda")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)])


def save_audio_to_local(audio):
    filename = 'temp.wav'
    if args.music_decoder == "audioldm2":
        scipy.io.wavfile.write(filename, rate=16000, data=audio[0])
    else:
        scipy.io.wavfile.write(filename, rate=model.generation_model.config.audio_encoder.sampling_rate, data=audio)
    return filename


def predict(
        prompt_input,
        audio_path,
        top_p=0.8,
        temperature=0.6):
    prompts = [llama.format_prompt(prompt_input)]
    prompts = [model.tokenizer(x).input_ids for x in prompts]
    image, audio, video = None, None, None
    if audio_path is not None:
        sample_rate = 24000
        waveform, sr = torchaudio.load(audio_path)
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)
        audio = torch.mean(waveform, 0)

    response = model.generate(prompts, audio, image, video, 512, temperature, top_p)
    return response[-1]['aud']


mtg = json.load(open("../../Datasets/MusicQA/MusicQA/EvalMusicQA.json"))

from tqdm import tqdm
from collections import defaultdict

m2ugen_data = defaultdict(lambda: {})
fileset = set()
out_filename = "./results/m2ugen_data.json"

if not os.path.exists("./results"):
    os.makedirs("./results")

if os.path.exists(out_filename):
    m2ugen_data = defaultdict(lambda: {}, json.load(open(out_filename, 'r')))
    fileset = set(m2ugen_data.keys())

print(f"Already Completed: {len(fileset)}")

count = 0

pbar = tqdm(total=total)

for row in mtg:
    if row["audio_name"] in fileset:
        continue
    audio = os.path.join("../../Datasets/MusicQA/MusicQA/audios", row["audio_name"])
    q = row["conversation"][0]["value"]
    m2ugen_data[row["audio_name"]][q] = predict(q, audio)
    pbar.update(1)
    count += 1
    if count % 10 == 0:
        with open(out_filename, 'w') as f:
            json.dump(m2ugen_data, f)

with open(out_filename, 'w') as f:
    json.dump(m2ugen_data, f)
