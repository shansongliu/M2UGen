import json
import sys
import os
from tqdm.auto import tqdm

sys.path.append('../common')

from mpt import get_qa

human_message = """<|im_start|>system
- You are given description of a music and an image
- You will give a single line instruction of the form 'Generate a music for the image that is ....' based on the image
- Complete the instruction based on the music and image description <|im_end|>
"""

gpt_message = """<|im_start|>system
- You are given description of a music and an image
- You will give a single line answer of the form 'Here is a music that is ....' based on the image
- Complete the answer based on the music and image description <|im_end|>
"""


def human_generate(caption):
    return get_qa(caption, instructions=human_message).split("\n")[-1]


def gpt_generate(caption):
    return get_qa(caption, instructions=gpt_message).split("\n")[-1]


image_captions = json.load(open("MUImageImageCaptions.json", "r"))
music_captions = json.load(open("MUImageMusicCaptions.json", "r"))

muimage_instructions = [] if not os.path.exists("MUImageInstructions.json") else json.load(open("MUImageInstructions.json", "r"))

count = 0

for name in tqdm(music_captions.keys()):
    music_caption, image_caption = music_captions[name], image_captions[name.replace(".mp3", ".jpg")]
    mpt_input = f"Music: {music_caption}\nImage: {image_caption}"
    conversation = [{
        "from": "human",
        "value": human_generate(mpt_input),
        "input_modality": "image",
        "caption": image_caption}, {
        "from": "gpt",
        "value": gpt_generate(mpt_input),
        "caption": music_caption,
        "output_modality": "audio"
    }]
    subdata = {"input_file": name, "output_file": name.replace(".mp3", ".jpg"), "conversation": conversation}
    muimage_instructions.append(subdata)
    count += 1
    if count % 10 == 0:
        json.dump(muimage_instructions, open("MUImageInstructions.json", "w"))
