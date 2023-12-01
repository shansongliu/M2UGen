from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    AutoConfig
)
import torch

model_name = "mosaicml/mpt-7b-chat"
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.attn_config['attn_impl'] = 'torch'
config.init_device = 'cuda'  # For fast initialization directly on GPU!
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, trust_remote_code=True,
                                             torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", trust_remote_code=True)

stop_token_ids = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])

start_message = """<|im_start|>system
- You are given descriptions describing an input and a target music
- The aim is to convert input music to the target
- You will give instructions of all the things to be added to, removed from or replaced in the input music
- Maximum number of instructions is 3 and do not use the words input or target
- Use only imperative sentences and make it a list <|im_end|>
"""


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


import re


def convert_history_to_text(history, sm=start_message):
    text = sm + "".join(
        [
            "".join(
                [
                    f"<|im_start|>user\n{item[0]}<|im_end|>",
                    f"<|im_start|>assistant\n{item[1]}<|im_end|>",
                ]
            )
            for item in history[:-1]
        ]
    )
    text += "".join(
        [
            "".join(
                [
                    f"<|im_start|>user\n{history[-1][0]}<|im_end|>",
                    f"<|im_start|>assistant\n{history[-1][1]}",
                ]
            )
        ]
    )
    return text


def bot(history, temperature=0.5, top_p=1, top_k=4, repetition_penalty=1, instructions=start_message):
    stop = StopOnTokens()

    # Construct the input message string for the model by concatenating the current system message and conversation
    # history
    messages = convert_history_to_text(history, sm=instructions)

    # Tokenize the messages string
    input_ids = tokenizer(messages, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    if input_ids.shape[-1] > 2040:
        return None
    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=8192,
        temperature=temperature,
        do_sample=temperature > 0.0,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        stopping_criteria=StoppingCriteriaList([stop]),
    )

    full_history = tokenizer.batch_decode(model.generate(**generate_kwargs), skip_special_tokens=True)[0]
    return full_history


def get_qa(caption, instructions=start_message):
    return bot([[caption, ""]], instructions=instructions)
