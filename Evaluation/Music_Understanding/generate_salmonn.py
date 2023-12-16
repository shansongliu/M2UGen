import sys

sys.path.append('../Models/SALMONN')

from model import SALMONN

model = SALMONN(
    ckpt="../Models/SALMONN/salmonn_v1.pth",
    whisper_path="../Models/SALMONN/whisper-large-v2",
    beats_path="../Models/SALMONN/BEATs_iter3.pt",
    vicuna_path="../Models/SALMONN/vicuna-13b-v1.1",
    low_resource=True
)

model.to("cuda")
model.eval()

import json

data = json.load(open("../../Datasets/MusicQA/EvalMusicQA.json", "r"))

root_dir = "../../Datasets/MusicQA/audios"


def test_row(row):
    result = model.generate(f"{root_dir}/{row['audio_name']}", prompt=row['conversation'][0]['value'])[0]
    expected = row['conversation'][-1]['value']
    return result, expected


from tqdm.auto import tqdm

results = []
count = 0

for row in tqdm(data):
    results.append(test_row(row))
    count += 1
    if count % 20 == 0:
        json.dump(results, open(f"./results/salmonn_data.json", "w"))

json.dump(results, open(f"./results/salmonn_data.json", "w"))
