from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score as meteor_scorer
from nltk.tokenize import wordpunct_tokenize
import json
from bert_score import score
from tqdm.auto import tqdm

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

mullama_data = json.load(open("./results/mullama_data.json", "r"))
ltu_data = json.load(open("./results/ltu_data.json", "r"))
llama_data = json.load(open("./results/llama-adapter_data.json", "r"))
salmonn_data = json.load(open("./results/salmonn_data.json", "r"))
m2ugen_data = json.load(open("./results/m2ugen_data.json", "r"))

mtg_data = json.load(open("../../Datasets/MusicQA/MusicQA/EvalMusicQA.json", "r"))


def evaluate(model_name, candidates, mult_reference):
    rouge_score, bleu_score, bleu4_score, meteor_score = 0, 0, 0, 0
    for ref, cand in tqdm(zip(mult_reference, candidates), total=len(mult_reference)):
        rouge_score += scorer.score(ref, cand)['rougeL'].recall
        cand_split = wordpunct_tokenize(cand)
        ref_split = wordpunct_tokenize(ref)
        bleu4_score += sentence_bleu([ref], cand, weights=(0.0, 0.0, 0.0, 1.0))
        bleu_score += sentence_bleu([ref], cand)
        meteor_score += meteor_scorer([ref_split], cand_split)
    rouge_score, bleu_score, bleu4_score, meteor_score = rouge_score / (len(candidates)), bleu_score / (
        len(candidates)), bleu4_score / (len(candidates)), meteor_score / (len(candidates))
    P, R, F1 = score(candidates, mult_reference, lang="en", verbose=True)
    bert_score = R.mean().item()
    print(f"Model: {model_name}")
    print(f"BLEU Score: {bleu_score}")
    print(f"BLEU-4 Score: {bleu4_score}")
    print(f"METEOR Score: {meteor_score}")
    print(f"ROUGE Score: {rouge_score}")
    print(f"BERT Score: {bert_score}")


reference = {"LTU": [], "LLaMA Adapter": [], "MU-LLaMA": [], "SALMONN": [], "M2UGen": []}
candidates = {"LTU": [], "LLaMA Adapter": [], "MU-LLaMA": [], "SALMONN": [], "M2UGen": []}

for row in tqdm(mtg_data):
    audio = row["audio_name"]
    if audio in ltu_data and row["conversation"][0]["value"] in ltu_data[audio]:
        candidates["LTU"].append(ltu_data[audio][row["conversation"][0]["value"]])
        reference["LTU"].append(row["conversation"][1]["value"])
    if audio in mullama_data and row["conversation"][0]["value"] in mullama_data[audio]:
        candidates["MU-LLaMA"].append(mullama_data[audio][row["conversation"][0]["value"]])
        reference["MU-LLaMA"].append(row["conversation"][1]["value"])
    if audio in llama_data and row["conversation"][0]["value"] in llama_data[audio]:
        candidates["LLaMA Adapter"].append(llama_data[audio][row["conversation"][0]["value"]])
        reference["LLaMA Adapter"].append(row["conversation"][1]["value"])
    if audio in salmonn_data and row["conversation"][0]["value"] in salmonn_data[audio]:
        candidates["SALMONN"].append(salmonn_data[audio][row["conversation"][0]["value"]])
        reference["SALMONN"].append(row["conversation"][1]["value"])
    if audio in m2ugen_data and row["conversation"][0]["value"] in m2ugen_data[audio]:
        candidates["M2UGen"].append(m2ugen_data[audio][row["conversation"][0]["value"]])
        reference["M2UGen"].append(row["conversation"][1]["value"])

for model, val in candidates.items():
    evaluate(model, val, reference[model])
