import sys

sys.path.append('../../M2UGen')

import json
from tqdm import tqdm
from llama.m2ugen import M2UGen
import llama
import os
from pydub import AudioSegment
import scipy
import torch
import torchvision.transforms as transforms
import argparse
import av
import numpy as np

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

data = json.load(open("../../Datasets/MUVideo/MUVideoEvalInstructions.json"))

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)])


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


def generate(prompt, video_file, length_in_sec=10, top_p=0.8, temperature=0.6):
    container = av.open(video_file)
    indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
    video = read_video_pyav(container=container, indices=indices)
    prompts = [llama.format_prompt(prompt)]
    prompts = [model.tokenizer(x).input_ids for x in prompts]
    audio, image = None, None
    response = model.generate(prompts, audio, image, video, 512,
                              temperature=temperature, top_p=top_p, audio_length_in_s=length_in_sec)
    return response[-1]['aud']


if not os.path.exists("./results/m2ugen"):
    os.makedirs("./results/m2ugen")

for row in tqdm(data):
    video = f"../../Datasets/MUVideo/audioset_videos/{row['input_file']}"
    music = row['output_file']
    prompt = row['conversation'][0]['value']
    audioSegment = AudioSegment.from_wav(os.path.join("../../Datasets/MUVideo/audioset", music))
    audio = generate(prompt, video, length_in_sec=audioSegment.duration_seconds)
    scipy.io.wavfile.write(f"./results/m2ugen/{music.replace('.mp3', '.wav')}", rate=16000, data=audio)
