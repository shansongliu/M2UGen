import sys

sys.path.append('../Models/CoDi')

import json
from tqdm import tqdm
from core.models.model_module_infer import model_module
import os
from pydub import AudioSegment
import scipy
import torch

data = json.load(open("../../Datasets/MUCaps/MUCapsEvalCaptions.json"))

model_load_paths = ['../Models/CoDi/checkpoints/CoDi_encoders.pth',
                    '../Models/CoDi/checkpoints/CoDi_text_diffuser.pth',
                    '../Models/CoDi/checkpoints/CoDi_audio_diffuser_m.pth',
                    '../Models/CoDi/checkpoints/CoDi_video_diffuser_8frames.pth']

model = model_module(data_dir='../Models/CoDi/checkpoints/', pth=model_load_paths, fp16=False)
model.eval()
model = model.to("cuda")


def generate(prompt, length_in_sec=10):
    audio_wave = model.inference(
        xtype=['audio'],
        condition=[prompt],
        condition_types=['text'],
        scale=7.5,
        n_samples=1,
        ddim_steps=50)[0]
    return audio_wave.squeeze()


if not os.path.exists("./results/codi"):
    os.makedirs("./results/codi")

for music, caption in tqdm(data):
    audioSegment = AudioSegment.from_wav(os.path.join("../../Datasets/MUCaps/audios_eval", music))
    audio = generate(caption, length_in_sec=audioSegment.duration_seconds)
    scipy.io.wavfile.write(f"./results/codi/{music.replace('.mp3', '.wav')}", rate=16000, data=audio)
