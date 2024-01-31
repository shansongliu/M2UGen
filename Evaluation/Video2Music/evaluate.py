import re
import subprocess
from hear21passt.base import get_basic_model, get_model_passt
import librosa
from sklearn.metrics import mutual_info_score
import torch
from pathlib import Path
import json
import sys

from tqdm import tqdm

sys.path.append('../Models/imagebind_LLM/ImageBind')

import data
from models import imagebind_model
from models.imagebind_model import ModalityType

models = ['CoDi', 'M2UGen_v1', 'M2UGen_v2']
scores = {model: {"FAD": 0, "CLAP": 0, "IM_RANK": 0} for model in models}
model_files = {"CoDi": "./results/codi", "M2UGen_v1": "./results/m2ugen_v1", "M2UGen_v2": "./results/m2ugen_v2"}
model_order = {k: v for k, v in enumerate(models)}

kl_model = get_basic_model(mode="logits")
kl_model.eval()
kl_model = kl_model.cuda()

imbind = imagebind_model.imagebind_huge(pretrained=True)
imbind.eval()
imbind.cuda()


def load_audio(filename):
    y, sr = librosa.load(filename, sr=32000)
    y = y[:32000 * 10]
    return torch.tensor(y).unsqueeze(0)


def compare_files(file1, file2):
    audio1 = load_audio(file1)
    audio2 = load_audio(file2)
    audio_wave1 = audio1.cuda()
    logits1 = kl_model(audio_wave1).cpu().detach()
    probs1 = torch.softmax(logits1, dim=-1)
    audio_wave2 = audio2.cuda()
    logits2 = kl_model(audio_wave2).cpu().detach()
    probs2 = torch.softmax(logits2, dim=-1)
    return probs1, probs2


def load_clap_audio(filename):
    y, _ = librosa.load(filename, sr=48000)
    y = y.reshape(1, -1)
    return y


def imbind_rank(video, m1, m2, c):
    inputs = {
        ModalityType.VISION: data.load_and_transform_video_data([video] * 3, "cuda"),
        ModalityType.AUDIO: data.load_and_transform_audio_data([m1, m2, c], "cuda")
    }
    with torch.no_grad():
        embeddings = imbind(inputs)
    cosine_similarity = torch.nn.functional.cosine_similarity(embeddings[ModalityType.AUDIO],
                                                              embeddings[ModalityType.TEXT],
                                                              dim=1, eps=1e-8)
    ranking = torch.argsort(cosine_similarity) + 1
    return {model: 1 / (3 - ranking[i] + 1) for i, model in enumerate(models)}


def kl_divergence(pred_probs: torch.Tensor, target_probs: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    kl_div = torch.nn.functional.kl_div((pred_probs + epsilon).log(), target_probs, reduction="none")
    return kl_div.sum(-1).mean()


def fad_score(original, generated):
    print("Calculating FAD Score...")
    command = f"fad_embed --verbose vggish {original} {generated}".split(" ")
    subprocess.run(command)
    command2 = f"fad_score {original}_emb_vggish {generated}_emb_vggish".split(" ")
    result2 = subprocess.run(command2, stdout=subprocess.PIPE)
    match = re.search("FAD score\s=\s+(\d*\.?\d*)", result2.stdout.decode())
    return float(match.group(1))


data = json.load(open("../../Datasets/MUVideo/MUVideoEvalInstructions.json"))

for model in scores.keys():
    scores[model]["FAD"] = fad_score("../../Datasets/MUVideo/audios_eval", model_files[model])
    generated_files = [str(x).split("/")[-1] for x in Path(model_files[model]).glob("*.mp3")]
    target_prob, pred_prob = [], []
    for music in generated_files:
        p1, p2 = compare_files(f"../../Datasets/MUVideo/audios_eval/{music}",
                               f"{model_files[model]}/{music}")
        target_prob.append(p1)
        pred_prob.append(p2)
    target_prob = torch.stack(target_prob, dim=0)
    pred_prob = torch.stack(pred_prob, dim=0)
    scores[model]["KL"] = kl_divergence(pred_prob, target_prob)

for row in tqdm(data):
    music = row['output_file']
    video = row['input_file']
    model_rankings = imbind_rank(video, f"{model_files['CoDi']}/{music}",
                                 f"{model_files['M2UGen_v1']}/{music}",
                                 f"{model_files['M2UGen_v2']}/{music}")
    for model, rank in model_rankings.items():
        scores[model]["IM_RANK"] += rank

for model in models:
    scores[model]["IM_RANK"] /= len(data)

print(json.dumps(scores, indent=4))
