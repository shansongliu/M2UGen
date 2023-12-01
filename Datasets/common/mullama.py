import sys

sys.path.append('../../MU-LLaMA')

import torch.cuda

import llama
from util.misc import *
from data.utils import load_and_transform_audio_data

model = llama.load("../../MU-LLaMA/ckpts/checkpoint.pth", "../../M2UGen/ckpts/LLaMA-2",
                   knn=True, knn_dir="../../M2UGen/ckpts", llama_type="7B")
model.eval()


def multimodal_generate(
        audio_path,
        audio_weight,
        prompt,
        cache_size,
        cache_t,
        cache_weight,
        max_gen_len,
        gen_t, top_p
):
    inputs = {}
    audio = load_and_transform_audio_data([audio_path])
    inputs['Audio'] = [audio, audio_weight]
    image_prompt = prompt
    text_output = None
    prompts = [llama.format_prompt(prompt)]
    prompts = [model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    with torch.cuda.amp.autocast():
        results = model.generate(inputs, prompts, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p,
                                 cache_size=cache_size, cache_t=cache_t, cache_weight=cache_weight)
    text_output = results[0].strip()
    return text_output


def qa_bot(audio, query="Describe the music"):
    return multimodal_generate(audio, 1, query, 100, 20.0, 0.3, 8192, 0.4, 1.0)
