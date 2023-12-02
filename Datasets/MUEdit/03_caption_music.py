import json
import sys
import os
from tqdm.auto import tqdm
import torch
sys.path.append('../common')

from mullama import qa_bot

pairs = set([tuple(x) for x in json.load(open("ValidPairs.json", "r"))])
fileset = set()

for a, b in pairs:
    fileset.add(a)
    fileset.add(b)

fileset = list(fileset)

caption_data = {}
done_fileset = set()

if os.path.exists(f"./MUEditCaptions.json"):
    caption_data = json.load(open(f'./MUEditCaptions.json', 'r'))
    done_fileset = set([f"audioset/" + x for x in caption_data.keys()])

count = 0

question = "Describe the music in detail including details such as instruments used, tempo and mood of the song"

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
        with open(f'./MUEditCaptions.json', 'w') as f:
            json.dump(caption_data, f)

with open(f'./MUEditCaptions.json', 'w') as f:
    json.dump(caption_data, f)
