import json
from tqdm import tqdm
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import os
import numpy as np
from pydub import AudioSegment
import scipy

data = json.load(open("../../Datasets/MUCaps/MUCapsEvalCaptions.json"))

processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")
model = model.to("cuda")


def generate(prompt, audio=None, length_in_sec=10):
    sampling_rate = model.config.audio_encoder.sampling_rate
    token_length = int(256*length_in_sec/5)
    start = 0
    audios = []
    while start < token_length:
        length = min(256*4, token_length - start)
        inputs = processor(
            audio=audio,
            sampling_rate=sampling_rate,
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        audio_values = model.generate(**inputs, guidance_scale=3.5, max_new_tokens=length)
        audios.append(audio_values[0].cpu().numpy()[0])
        audio = audios[-1][:10*sampling_rate]
        start += length
    audio = np.concatenate(audios, axis=0)
    return audio


if not os.path.exists("./results/musicgen"):
    os.makedirs("./results/musicgen")

for music, caption in tqdm(data):
    audioSegment = AudioSegment.from_wav(os.path.join("../../Datasets/MUCaps/audios_eval", music))
    audio = generate(caption, length_in_sec=audioSegment.duration_seconds)
    scipy.io.wavfile.write(f"./results/musicgen/{music.replace('.mp3', '.wav')}", rate=model.config.audio_encoder.sampling_rate, data=audio)
