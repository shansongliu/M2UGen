import json
import sys
import os
from tqdm.auto import tqdm

sys.path.append('../common')

from mpt import get_qa

human_message = """<|im_start|>system
- You are given description of input and target music
- You will give a single line instruction of the form to change input music to target music
- Give the instruction based on the music descriptions <|im_end|>
"""

gpt_message = """<|im_start|>system
- You are given description of a music
- You will give a single line answer of the form 'Here is a music that is ....'
- Complete the answer based on the music description <|im_end|>
"""


def human_generate(caption):
    answer = get_qa(caption, instructions=human_message)
    if answer is None:
        return None
    return answer.split("\n")[-1]


def gpt_generate(caption):
    answer = get_qa(caption, instructions=gpt_message)
    if answer is None:
        return None
    return answer.split("\n")[-1]


pairs = set([tuple(x) for x in json.load(open("ValidPairs.json", "r"))])
music_captions = json.load(open(f'./MuEditCaptions.json', 'r'))

muedit_instructions = [] if not os.path.exists(f"./MUEditInstructions.json") else json.load(open(f"./MUEditInstructions.json", "r"))

for f1, f2 in tqdm(pairs):
    f1 = f1.split('/')[-1]
    f2 = f2.split('/')[-1]
    music_caption1, music_caption2 = music_captions[f1], music_captions[f2]
    mpt_input = f"Input: {music_caption1}, Target: {music_caption2}"
    conversation = [{
        "from": "human",
        "value": human_generate(mpt_input),
        "input_modality": "audio",
        "caption": music_caption1}, {
        "from": "gpt",
        "value": gpt_generate(mpt_input),
        "caption": music_caption2,
        "output_modality": "audio"
    }]
    subdata = {"input_file": f1, "output_file": f2, "conversation": conversation}
    muedit_instructions.append(subdata)
