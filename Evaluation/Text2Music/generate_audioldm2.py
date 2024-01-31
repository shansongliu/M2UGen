import json
from tqdm import tqdm
from diffusers import AudioLDM2Pipeline
import os
from pydub import AudioSegment
import scipy
import torch

data = json.load(open("../../Datasets/MUCaps/MUCapsEvalCaptions.json"))

model = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", torch_dtype=torch.float16)
model = model.to("cuda")


def generate(prompt, length_in_sec=10):
    audio = model(
        prompt,
        negative_prompt='Low Quality',
        num_inference_steps=200,
        audio_length_in_s=length_in_sec,
        num_waveforms_per_prompt=3
    ).audios
    return audio


if not os.path.exists("./results/audioldm2"):
    os.makedirs("./results/audioldm2")

for music, caption in tqdm(data):
    audioSegment = AudioSegment.from_wav(os.path.join("../../Datasets/MUCaps/audios_eval", music))
    audio = generate(caption, length_in_sec=audioSegment.duration_seconds)
    scipy.io.wavfile.write(f"./results/audioldm2/{music.replace('.mp3', '.wav')}", rate=16000, data=audio[0])
