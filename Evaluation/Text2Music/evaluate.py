from frechet_audio_distance import FrechetAudioDistance
from hear21passt.base import get_basic_model, get_model_passt
import librosa
from sklearn.metrics import mutual_info_score
import torch
import laion_clap
from pathlib import Path
import json

scores = {model: {"FAD": 0, "CLAP": 0, "KL": 0} for model in ['MusicGen', 'AudioLDM2', 'CoDi', 'M2UGen']}
model_files = {"MusicGen": "./results/musicgen", "AudioLDM2": "./results/audioldm2", "CoDi": "./results/codi",
               "M2UGen": "./results/m2ugen"}

frechet = FrechetAudioDistance(model_name="vggish",
                               use_pca=False,
                               use_activation=False,
                               verbose=True)

kl_model = get_basic_model(mode="logits")
kl_model.eval()
kl_model = kl_model.cuda()

clap = laion_clap.CLAP_Module(enable_fusion=False)
clap.load_ckpt()


def load_audio(filename):
    y, sr = librosa.load(filename, sr=32000)
    y = y[:32000 * 10]
    return torch.tensor(y).unsqueeze(0)


def compare_files(file1, file2):
    audio1 = load_audio(file1)
    audio2 = load_audio(file2)
    audio_wave1 = audio1.cuda()
    logits1 = kl_model(audio_wave1).cpu().detach().numpy()
    audio_wave2 = audio2.cuda()
    logits2 = kl_model(audio_wave2).cpu().detach().numpy()
    return mutual_info_score(logits1[0], logits2[0])


def load_clap_audio(filename):
    y, _ = librosa.load(filename, sr=48000)
    y = y.reshape(1, -1)
    return y


def clap_compare(filename, caption):
    a = load_clap_audio(filename)
    ae = clap.get_audio_embedding_from_data(x=a)
    te = clap.get_text_embedding([caption])
    return torch.nn.functional.cosine_similarity(ae, te, dim=1, eps=1e-8)


data = json.load(open("../../Datasets/MUCaps/MUCapsEvalCaptions.json"))

for model in scores.keys():
    scores[model]["FAD"] = frechet.score("../../Datasets/MUCaps/audios_eval", model_files[model])
    generated_files = [str(x).split("/")[-1] for x in Path(model_files[model]).glob("*.mp3")]
    for music in generated_files:
        scores[model]["KL"] += compare_files(f"../../Datasets/MUCaps/audios_eval/{music}",
                                             f"{model_files[model]}/{music}")
    scores[model]["KL"] /= len(generated_files)
    for music, caption in data.items():
        scores[model]["CLAP"] = clap_compare(f"{model_files[model]}/{music}", caption)
    scores[model]["CLAP"] /= len(data)

print(json.dumps(scores, indent=4))
