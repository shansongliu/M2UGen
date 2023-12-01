from pathlib import Path
import random
from tqdm.auto import tqdm
import librosa
import numpy as np
import json


def compare(file1, file2):
    a1, _ = librosa.load(file1, sr=32000)
    a2, _ = librosa.load(file2, sr=32000)
    t1, b1 = librosa.beat.beat_track(y=a1, sr=32000)
    t2, b2 = librosa.beat.beat_track(y=a2, sr=32000)
    tempo_diff = abs(t1 - t2)
    return (np.square(a1 - a2).mean().sqrt().item() + tempo_diff + np.square(b1 - b2).mean().sqrt().item()) / 3


files = sorted([str(x) for x in Path("./audioset").glob("*.mp3")])
pairs = set() if not os.path.exists("ValidPairs.json") else set([tuple(x) for x in json.load(open("ValidPairs.json", "r"))])
count = 0

while count < 10000:
    a, b = random.randint(0, len(files)), random.randint(0, len(files))
    file1, file2 = files[a], files[b]
    if a == b or (a, b) in pairs:
        continue
    score = compare(files[a], files[b])
    if score > 1:
        continue
    pairs.add((files[a], files[b]))
    count += 1
    if count % 10 == 0:
        json.dump(list(pairs), open("ValidPairs.json", "w"))
