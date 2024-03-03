import pandas as pd
import json
from tqdm.auto import tqdm
from yt_dlp import YoutubeDL
import multiprocessing as mp
import datetime as dt
import os
import cv2
import random


def get_frame(video):
    cam = cv2.VideoCapture(video)
    frames = []
    while True:
        ret, frame = cam.read()
        if ret:
            frames.append(frame)
        else:
            break
    cam.release()
    if len(frames) == 0:
        return False
    frame = random.choice(frames)
    filename = video.replace(".mp4", ".jpg")
    cv2.imwrite(filename, frame)
    os.remove(video)
    return True


def _download_audio_video(x):
    (
        ytid,
        start,
        end,
        out_dir,
    ) = x
    if os.path.exists(f"{out_dir}/{ytid}.mp3"):
        return
    start_dt, end_dt = dt.timedelta(milliseconds=start), dt.timedelta(milliseconds=end)
    ydl_opts = {
        "outtmpl": f"{out_dir}/{ytid}.%(ext)s",
        "format": "bestaudio[ext=webm]/bestaudio/best",
        "external_downloader": "ffmpeg",
        "external_downloader_args": ["-ss", str(start_dt), "-to", str(end_dt)],
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }
        ],
        "quiet": True,
        "no-mtime": True,
    }
    ydl_video_opts = {
        "outtmpl": f"{out_dir}_images/{ytid}.%(ext)s",
        'format_sort': ['res:1080', 'ext:mp4'],
        "external_downloader": "ffmpeg",
        "external_downloader_args": ["-ss", str(start_dt), "-to", str(end_dt)],
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",  # one of avi, flv, mkv, mp4, ogg, webm
            }
        ],
        "quiet": True,
        "no-mtime": True,
    }
    try:
        yturl = f"https://www.youtube.com/watch?v={ytid}"
        with YoutubeDL(ydl_video_opts) as ydl:
            ydl.download([yturl])
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([yturl])
        if not get_frame(f"{out_dir}_images/{ytid}.mp4"):
            os.remove(f"{out_dir}/{ytid}.mp3")
    except KeyboardInterrupt:
        raise
    except Exception:
        pass


def download_ps(ytid, starts, ends, save_path, num_processes=None, desc=None):
    with mp.Pool(processes=num_processes) as pool, tqdm(total=len(ytid), desc=desc) as pbar:
        for _ in tqdm(
                pool.imap(
                    _download_audio_video,
                    zip(ytid, starts, ends, [save_path] * len(ytid)),
                ),
                total=len(ytid),
        ):
            pbar.update()


if __name__ == "__main__":
    df = pd.read_csv("../common/filtered.csv")
    print(f"Total Count: {len(df['YTID'].unique())}")
    download_ps(df["YTID"][30000:40000],
                [x * 1000 for x in df["start_seconds"][30000:40000]],
                [x * 1000 for x in df["end_seconds"][30000:40000]], "./audioset")
