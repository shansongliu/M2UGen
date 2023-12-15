import torch.cuda
import sys

sys.path.append('../Models/MU-LLaMA')

import llama
from util.misc import *
from data.utils import load_and_transform_audio_data
import json
import os
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="../Models/MU-LLaMA/MU-LLaMA/ckpts/music_finetune/checkpoint.pth", type=str,
    help="Name of or path to the trained checkpoint",
)
parser.add_argument(
    "--knn", default="../Models/MU-LLaMA/MU-LLaMA/ckpts", type=str,
    help="Name of or path to the directory with knn checkpoint",
)
parser.add_argument(
    "--llama_type", default="7B", type=str,
    help="Type of llama original weight",
)
parser.add_argument(
    "--llama_dir", default="../Models/MU-LLaMA/MU-LLaMA/ckpts/LLaMA-2", type=str,
    help="Path to LLaMA pretrained checkpoint",
)
parser.add_argument(
    "--mert_path", default="m-a-p/MERT-v1-330M", type=str,
    help="Path to MERT pretrained checkpoint",
)
args = parser.parse_args()

model = llama.load(args.model, args.llama_dir, mert_path=args.mert_path, knn=True,
                   knn_dir=args.knn, llama_type=args.llama_type)
model.eval()


def multimodal_generate(
        audio_path,
        audio_weight,
        prompt,
        cache_size,
        cache_t,
        cache_weight,
        max_gen_len,
        gen_t, top_p
):
    inputs = {}
    audio = load_and_transform_audio_data([audio_path])
    inputs['Audio'] = [audio, audio_weight]
    image_prompt = prompt
    text_output = None
    prompts = [llama.format_prompt(prompt)]
    prompts = [model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    with torch.cuda.amp.autocast():
        results = model.generate(inputs, prompts, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p,
                                 cache_size=cache_size, cache_t=cache_t, cache_weight=cache_weight)
    text_output = results[0].strip()
    return text_output


def qa_bot(audio, query="Describe the music"):
    return multimodal_generate(audio, 1, query, 100, 20.0, 0.2, 1024, 0.6, 0.8)


import librosa
from pydub import AudioSegment


def split_audio(audio, i):
    if len(audio) > 60000:
        audio[:60000].export("./ltemp_" + str(i) + ".wav", format="wav")
        return ["./ltemp_" + str(i) + ".wav"] + split_audio(audio[60000:], i + 1)
    if len(audio) > 10000:
        audio.export("./ltemp_" + str(i) + ".wav", format="wav")
        return ["./temp_" + str(i) + ".wav"]
    return []


def get_duration(filename):
    return librosa.get_duration(path=filename)


def get_split_count(duration):
    if duration > 60:
        return 1 + get_split_count(duration - 60)
    if duration > 10:
        return 1
    return 0


mtg = json.load(open("../../Datasets/MusicQA/MusicQA/EvalMusicQA.json"))

total = 0

for row in mtg:
    duration = get_duration(os.path.join("../../Datasets/MusicQA/MusicQA/audios", row["audio_name"]))
    total += get_split_count(duration) // 2

from tqdm import tqdm

mullama_data = defaultdict(lambda: {})
fileset = set()
out_filename = "./results/mullama_data.json"

if not os.path.exists("./results"):
    os.makedirs("./results")

if os.path.exists(out_filename):
    mullama_data = defaultdict(lambda: {}, json.load(open(out_filename, 'r')))
    fileset = set(mullama_data.keys())

print(f"Already Completed: {len(fileset)}")

count = 0

pbar = tqdm(total=total)

for row in mtg:
    if row["audio_name"] in fileset:
        continue
    audio = AudioSegment.from_wav(os.path.join("../../Datasets/MusicQA/MusicQA/audios", row["audio_name"]))
    audio_splits = split_audio(audio, 1)
    q = row["conversation"][0]["value"]
    result = []
    for j in range(0, len(audio_splits), 2):
        audio = audio_splits[j]
        result.append(qa_bot(audio, query=q))
        pbar.update(1)
    mullama_data[row["audio_name"]][q] = " ".join(result)
    count += 1
    if count % 10 == 0:
        with open(out_filename, 'w') as f:
            json.dump(mullama_data, f)

with open(out_filename, 'w') as f:
    json.dump(mullama_data, f)