import sys
import torch.cuda
import os
from pathlib import Path
from tqdm.auto import tqdm
import json
import librosa
from transformers import BertTokenizer

sys.path.append('../../MU-LLaMA')

import llama
from util.misc import *
from data.utils import load_and_transform_audio_data

model = llama.load("../../MU-LLaMA/ckpts/checkpoint.pth",
                   "../../M2UGen/ckpts/LLaMA-2",
                   knn=True, knn_dir="../../M2UGen/ckpts/knn.index", llama_type="7B")
model.eval()

bert = BertTokenizer("bert-base-uncased")


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
    prompts = [llama.format_prompt(prompt)]
    prompts = [model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    with torch.cuda.amp.autocast():
        text_output = model.generate(inputs, prompts, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p,
                                     cache_size=cache_size, cache_t=cache_t, cache_weight=cache_weight)
    text_output = text_output[0].strip()
    return text_output


fileset = sorted([str(x) for x in Path(r'./MUCaps/audios').glob('*.mp3')])


def get_duration(filename):
    return librosa.get_duration(path=filename)


def get_split_count(duration):
    if duration > 60:
        return 1 + get_split_count(duration - 60)
    return 0


def split_audio(audio, i):
    if len(audio) > 60000:
        audio[:60000].export(f"temp_" + str(i) + ".mp3", format="mp3")
        return [f"temp_" + str(i) + ".mp3"] + split_audio(audio[60000:], i + 1)
    if len(audio) > 10000:
        audio.export(f"temp_" + str(i) + ".mp3", format="mp3")
        return [f"temp_" + str(i) + ".mp3"]
    return []


def qa_bot(audio_path, query="Describe the music"):
    return multimodal_generate(audio_path, 1, query, 100, 20.0, 0.2, 1024, 0.2, 0.95)


caption_data = {}
done_fileset = set()

if os.path.exists(f"./MUCaps/MuCapsCaptions.json"):
    caption_data = json.load(open(f'./MUCaps/MuCapsCaptions.json', 'r'))
    done_fileset = set([f"MUCaps/audios/" + x for x in caption_data.keys()])

count = 0

question = "Describe the music in detail including details such as instruments used, tempo and mood of the song"
nq = 1

pbar = tqdm(total=len(fileset))

for file in fileset:
    torch.cuda.empty_cache()
    if file in done_fileset or not os.path.exists(file):
        pbar.update(1)
        continue
    caption = qa_bot(file, query=question)
    caption_data[file.split('/')[-1]] = caption
    count += 1
    if count % 10 == 0:
        with open(f'./MuCapsCaptions.json', 'w') as f:
            json.dump(caption_data, f)

with open(f'./MuCapsCaptions.json', 'w') as f:
    json.dump(caption_data, f)
