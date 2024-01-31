import sys

sys.path.append('../Models/CoDi')

import json
from tqdm import tqdm
from core.models.model_module_infer import model_module
from core.common.utils import load_video
import os
from pydub import AudioSegment
import scipy
import torch

data = json.load(open("../../Datasets/MUVideo/MUVideoEvalInstructions.json"))

model_load_paths = ['../Models/CoDi/checkpoints/CoDi_encoders.pth',
                    '../Models/CoDi/checkpoints/CoDi_text_diffuser.pth',
                    '../Models/CoDi/checkpoints/CoDi_audio_diffuser_m.pth',
                    '../Models/CoDi/checkpoints/CoDi_video_diffuser_8frames.pth']

model = model_module(data_dir='../Models/CoDi/checkpoints/', pth=model_load_paths, fp16=False)
model.eval()
model = model.to("cuda")


def generate(prompt, video_file):
    video = load_video(video_file, sample_duration=10.0, num_frames=8)
    audio_wave = model.inference(
        xtype=['audio'],
        condition=[prompt, video],
        condition_types=['text', 'video'],
        scale=7.5,
        n_samples=1,
        ddim_steps=50)[0]
    return audio_wave.squeeze()


if not os.path.exists("./results/codi"):
    os.makedirs("./results/codi")

for row in tqdm(data):
    video = f"../../Datasets/MUVideo/audioset_videos/{row['input_file']}"
    music = row['output_file']
    prompt = row['conversation'][0]['value']
    audioSegment = AudioSegment.from_wav(os.path.join("../../Datasets/MUVideo/audioset", music))
    audio = generate(prompt, video)
    scipy.io.wavfile.write(f"./results/codi/{music.replace('.mp3', '.wav')}", rate=16000, data=audio)
