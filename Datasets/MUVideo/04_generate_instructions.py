import json
import sys
import os
from tqdm.auto import tqdm

sys.path.append('../common')

from mpt import get_qa

human_message = """<|im_start|>system
- You are given description of a music and a video
- You will give a single line instruction of the form 'Generate a music for the video that is ....'
- Complete the instruction based on the music and video descriptions <|im_end|>
"""

gpt_message = """<|im_start|>system
- You are given description of a music and a video
- You will give a single line answer of the form 'Here is a music that is ....'
- Complete the answer based on the music and video descriptions <|im_end|>
"""


def human_generate(caption):
    return get_qa(caption, instructions=human_message).split("\n")[-1]


def gpt_generate(caption):
    return get_qa(caption, instructions=gpt_message).split("\n")[-1]


video_captions = json.load(open("MUVideoVideoCaptions.json", "r"))
music_captions = json.load(open("MUVideoMusicCaptions.json", "r"))

muvideo_instructions = [] if not os.path.exists("MUVideoInstructions.json") else json.load(open("MUVideoInstructions.json", "r"))

count = 0

for name in tqdm(music_captions.keys()):
    music_caption, video_caption = music_captions[name], video_captions[name.replace(".mp3", ".mp4")]
    mpt_input = f"Music: {music_caption}\nVideo: {video_caption}"
    conversation = [{
        "from": "human",
        "value": human_generate(mpt_input),
        "input_modality": "video",
        "caption": video_caption}, {
        "from": "gpt",
        "value": gpt_generate(mpt_input),
        "caption": music_caption,
        "output_modality": "audio"
    }]
    subdata = {"input_file": name.replace(".mp3", ".mp4"), "output_file": name, "conversation": conversation}
    muvideo_instructions.append(subdata)
    count += 1
    if count % 10 == 0:
        json.dump(muvideo_instructions, open("MUVideoInstructions.json", "w"))
