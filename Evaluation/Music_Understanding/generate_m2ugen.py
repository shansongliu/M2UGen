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


def parse_text(text, image_path, video_path, audio_path):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    outputs = text
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines) + "<br>"
    if image_path is not None:
        text += f'<img src="./file={image_path}" style="display: inline-block;"><br>'
        outputs = f'<Image>{image_path}</Image> ' + outputs
    if video_path is not None:
        text += f' <video controls playsinline height="320" width="240" style="display: inline-block;"  src="./file={video_path}"></video6><br>'
        outputs = f'<Video>{video_path}</Video> ' + outputs
    if audio_path is not None:
        text += f'<audio controls playsinline><source src="./file={audio_path}" type="audio/wav"></audio><br>'
        outputs = f'<Audio>{audio_path}</Audio> ' + outputs
    # text = text[::-1].replace(">rb<", "", 1)[::-1]
    text = text[:-len("<br>")].rstrip() if text.endswith("<br>") else text
    return text, outputs


def save_audio_to_local(audio):
    filename = 'temp.wav'
    if args.music_decoder == "audioldm2":
        scipy.io.wavfile.write(filename, rate=16000, data=audio[0])
    else:
        scipy.io.wavfile.write(filename, rate=model.generation_model.config.audio_encoder.sampling_rate, data=audio)
    return filename


def parse_reponse(model_outputs):
    response = ''
    text_outputs = []
    filename = None
    for output_i, p in enumerate(model_outputs):
        if isinstance(p, str):
            response += p
            response += '<br>'
            text_outputs.append(p)
        elif 'aud' in p.keys():
            _temp_output = ''
            for idx, m in enumerate(p['aud']):
                if isinstance(m, str):
                    response += m.replace(''.join([f'[AUD{i}]' for i in range(8)]), '')
                    response += '<br>'
                    _temp_output += m.replace(''.join([f'[AUD{i}]' for i in range(8)]), '')
                else:
                    filename = save_audio_to_local(m)
                    _temp_output = f'<Audio>{filename}</Audio> ' + _temp_output
                    response += f'<audio controls playsinline><source src="./file={filename}" type="audio/wav"></audio>'
            text_outputs.append(_temp_output)
        else:
            pass
    response = response[:-len("<br>")].rstrip() if response.endswith("<br>") else response
    return response, text_outputs, filename


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    if converted_len > seg_len:
        converted_len = 0
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def get_video_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return int(round(float(result.stdout)))


def get_audio_length(filename):
    return int(round(librosa.get_duration(path=filename)))


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
    return response[0]


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