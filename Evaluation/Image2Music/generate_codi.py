import sys

sys.path.append('../Models/CoDi')

import json
from tqdm import tqdm
from core.models.model_module_infer import model_module
from PIL import Image
import os
from pydub import AudioSegment
import scipy
import torch

data = json.load(open("../../Datasets/MUImage/MUImageEvalInstructions.json"))

model_load_paths = ['../Models/CoDi/checkpoints/CoDi_encoders.pth',
                    '../Models/CoDi/checkpoints/CoDi_text_diffuser.pth',
                    '../Models/CoDi/checkpoints/CoDi_audio_diffuser_m.pth',
                    '../Models/CoDi/checkpoints/CoDi_video_diffuser_8frames.pth']

model = model_module(data_dir='../Models/CoDi/checkpoints/', pth=model_load_paths, fp16=False)
model.eval()
model = model.to("cuda")


def generate(prompt, image_file):
    image = Image.open(image_file)
    audio_wave = model.inference(
        xtype=['audio'],
        condition=[prompt, image],
        condition_types=['text', 'image'],
        scale=7.5,
        n_samples=1,
        ddim_steps=50)[0]
    return audio_wave.squeeze()


if not os.path.exists("./results/codi"):
    os.makedirs("./results/codi")

for row in tqdm(data):
    image = f"../../Datasets/MUImage/audioset_images/{row['input_file']}"
    music = row['output_file']
    prompt = row['conversation'][0]['value']
    audioSegment = AudioSegment.from_wav(os.path.join("../../Datasets/MUImage/audioset", music))
    audio = generate(prompt, image)
    scipy.io.wavfile.write(f"./results/codi/{music.replace('.mp3', '.wav')}", rate=16000, data=audio)
