import os
import json

import requests
import base64
import re
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--demo', help='Link to the LTU Demo Page', default="https://6654419aa961b664df.gradio.live")
args = parser.parse_args()

mtg = json.load(open("../../Datasets/MusicQA/EvalMusicQA.json"))


def qa_bot(filename, query="Describe the music"):
    while True:
        with open(filename, "rb") as f:
            encoded_f1 = base64.b64encode(f.read())
            encoded_f1 = f"data:audio/mpeg;base64," + str(encoded_f1, 'ascii', 'ignore')
            r = requests.post(f'{args.demo}/run/predict',
                              json={"data": [{"data": encoded_f1, "name": filename}, query], "fn_index": 0,
                                    "session_hash": "xb8xaznbhy"})
            caption = r.json()["data"][1]
            caption = re.search(r"Response:\n(.*)$", caption)
            if caption is None:
                continue
            return caption.group(1)


if not os.path.exists("./results"):
    os.makedirs("./results")

from tqdm import tqdm

ltu_data = defaultdict(lambda: {})
fileset = set()
out_filename = "./results/ltu_data.json"

if os.path.exists(out_filename):
    ltu_data = defaultdict(lambda: {}, json.load(open(out_filename, 'r')))
    fileset = set(ltu_data.keys())

count = 0

for row in tqdm(mtg, total=len(mtg)):
    if row["audio_name"] in fileset:
        continue
    q = row["conversation"][0]["value"]
    a = qa_bot(os.path.join("../../Datasets/MusicQA/audios", row["audio_name"]), query=q)
    ltu_data[row["audio_name"]][q] = a
    count += 1
    if count % 10 == 0:
        with open(out_filename, 'w') as f:
            json.dump(ltu_data, f)

with open(out_filename, 'w') as f:
    json.dump(ltu_data, f)