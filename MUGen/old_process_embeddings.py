
import numpy as np
import os
import sys
from joblib import Parallel, delayed
from tqdm import tqdm
import torch
import json

# Load a slightly modified version of the Stable Diffusion pipeline.
# This allows us to extract text embeddings directly (without generating images).
from model.custom_ad import AudioLDMPipeline



def save_to_path(emb, path):
    """Save embeddings to disk."""
    try:
        with open(path, 'wb') as wf:
            np.save(wf, emb)
    except:
        print("Error with", path)
    return path


if __name__ == '__main__':

    batch_size = 128

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # clip_output_dir = './embed/'
    # synthesize_path = '../data/synthesize_data/synthesize_data.json'

    # video_path = '../data/T-X_pair_data/webvid/webvid.json'
    # audio_path = '../data/T-X_pair_data/audiocap/audiocap.json'
    # img_path = '../data/T-X_pair_data/cc3m/cc3m.json'

    # image_generation_ckpt_path = 'runwayml/stable-diffusion-v1-5'
    # video_generation_ckpt_path = 'cerspense/zeroscope_v2_576w'
    # audio_generation_ckpt_path = 'cvssp/audioldm-l-full'

    data_path = sys.argv[1]
    data_dir = sys.argv[2]
    clip_output_dir = sys.argv[3]

    if not os.path.exists(clip_output_dir):
        os.makedirs(clip_output_dir, exist_ok=True)

    # Get existing files, so that we don't recompute them.
    existing_files = set([f.split("/")[-1].strip('.npy') for f in os.listdir(clip_output_dir)])

    caption_list = []
    name_list = []
    print('extract audio caption embedding')
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if type(data) == dict:
        for one_audio_name, one_caption in tqdm(data.items(), total=len(data)):
            if one_audio_name not in existing_files:
                caption_list.append(one_caption)
                name_list.append(os.path.join(data_dir, one_audio_name))
    else:
        for row in tqdm(data):
            one_audio_name, one_caption = row["output_file"], row["conversation"][-1]["caption"]
            if one_audio_name not in existing_files:
                caption_list.append(one_caption)
                name_list.append(os.path.join(data_dir, one_audio_name))
    pipe = AudioLDMPipeline.from_pretrained("/hpctmp/e0589920/audioldm-l-full", torch_dtype=dtype)
    pipe = pipe.to("cuda")

    print('Extract embeddings in batches.')
    num_batches = int(np.ceil(len(caption_list) / batch_size))
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_captions = caption_list[start_idx:end_idx]
        batch_ids = name_list[start_idx:end_idx]
        prompt_embeds = pipe(batch_captions, return_prompts_only=True).detach().cpu().numpy()

        # Save embeddings to disk in parallel.
        Parallel(n_jobs=8)(delayed(save_to_path)(
            prompt_embeds[j, :, ...], os.path.join(clip_output_dir, f'{batch_ids[j].split("/")[-1]}.npy')
        ) for j in range(prompt_embeds.shape[0]))
