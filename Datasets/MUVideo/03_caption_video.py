import av
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
from pathlib import Path
import json
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# load pretrained processor, tokenizer, and model
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)


def videocaption(filename):
    container = av.open(filename)
    # extract evenly spaced frames from video
    seg_len = container.streams.video[0].frames
    clip_len = model.config.encoder.num_frames
    indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    # generate caption
    gen_kwargs = {
        "min_length": 20,
        "max_length": 200,
        "num_beams": 8,
    }
    pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
    tokens = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    return caption


video_files = [str(x) for x in Path("audioset_videos").glob("*.mp4")]

muimage_captions = {video.split("/")[-1]: videocaption(video) for video in tqdm(video_files)}
json.dump(muimage_captions, open("MUVideoVideoCaptions.json", "w"))
