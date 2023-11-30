import pandas as pd
import json
from tqdm.auto import tqdm
from yt_dlp import YoutubeDL
import multiprocessing as mp
import datetime as dt
import os

def _download_audio(x):
    (
        ytid,
        out_dir,
    ) = x
    if os.path.exists(f"{out_dir}/{ytid}.mp3"):
        return
    ydl_opts = {
        "outtmpl": f"{out_dir}/{ytid}.%(ext)s",
        "format": "bestaudio[ext=webm]/bestaudio/best",
        "external_downloader": "ffmpeg",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }
        ],
        "quiet": True,
        "no-mtime": True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={ytid}"])
    except KeyboardInterrupt:
        raise
    except Exception:
        pass


def download_ps(ytid, save_path, num_processes=None, desc=None):
    with mp.Pool(processes=num_processes) as pool, tqdm(total=len(ytid), desc=desc) as pbar:
        for _ in tqdm(
                pool.imap(
                    _download_audio,
                    zip(ytid, [save_path] * len(ytid)),
                ),
                total=len(ytid),
        ):
            pbar.update()


if __name__ == "__main__":
    ontologies = json.load(open('ontology.json', 'r'))

    ont_data = {}

    for ont in ontologies:
        ont_data[ont["id"]] = ont["name"].lower()

    df = pd.read_csv("audioset.csv")

    count = 0
    ids = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        valid = False
        for idx in row["positive_labels"].split(","):
            if "musical instrument" in ont_data[idx]:
                valid = True
            if "human" in ont_data[idx] \
                    or "conversation" in ont_data[idx] \
                    or "sing" in ont_data[idx] \
                    or "vocal" in ont_data[idx]:
                valid = False
        if valid:
            count += 1
            ids.append(i)

    df = df.iloc[ids]
    print(f"Total Count: {len(df['YTID'].unique())}")
    download_ps(df["YTID"].unique()[:20000], "./audioset_full")
