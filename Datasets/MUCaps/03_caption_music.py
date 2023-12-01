import sys
import torch.cuda
import os
from pathlib import Path
from tqdm.auto import tqdm
import json

sys.path.append('../common')

from mullama import qa_bot

model = llama.load("../../MU-LLaMA/ckpts/checkpoint.pth",
                   "../../M2UGen/ckpts/LLaMA-2",
                   knn=True, knn_dir="../../M2UGen/ckpts/knn.index", llama_type="7B")
model.eval()

fileset = sorted([str(x) for x in Path(r'./audioset_full').glob('*.mp3')])
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
