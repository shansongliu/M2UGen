import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import List

from transformers import Wav2Vec2FeatureExtractor, AutoModel


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def load_and_transform_audio_data(
        audio_paths,
        sample_rate=24000
) -> List[torch.Tensor]:
    audios = []
    for path in audio_paths:
        waveform, sr = torchaudio.load(path)
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)
        waveform = torch.mean(waveform, 0)
        audios.append(waveform)
    return audios


class MERTEncoder(nn.Module):

    def __init__(self, mert_path="/hpctmp/e0589920/MERT-v1-330M"):
        super().__init__()
        print(f'Initializing MERT encoder from {mert_path} ...')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mert_model = AutoModel.from_pretrained(mert_path, trust_remote_code=True).to(self.device)
        self.mert_model.eval()
        self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(mert_path, trust_remote_code=True)
        self.mu_mert_agg = nn.Conv1d(in_channels=25, out_channels=1, kernel_size=1)
        self.mu_mert_proj = nn.Linear(1024, 4096)

        bridge_norm_layer = RMSNorm
        bridge_bias = False

        self.mu_mert_norm_1 = bridge_norm_layer(4096)
        self.mu_mert_f1_1 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)
        self.mu_mert_f2_1 = nn.Linear(4096 * 4, 4096, bias=bridge_bias)
        self.mu_mert_f3_1 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)

        self.mu_mert_norm_2 = bridge_norm_layer(4096)
        self.mu_mert_f1_2 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)
        self.mu_mert_f2_2 = nn.Linear(4096 * 4, 4096, bias=bridge_bias)
        self.mu_mert_f3_2 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)

        self.mu_mert_norm_3 = bridge_norm_layer(4096)
        self.mu_mert_f1_3 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)
        self.mu_mert_f2_3 = nn.Linear(4096 * 4, 4096, bias=bridge_bias)
        self.mu_mert_f3_3 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)
        print('MERT encoder initialized.')

    def encode_audio(self, x: List[torch.Tensor]) -> torch.Tensor:
        xs = []
        for sub_x in x:
            all_inputs = [self.mert_processor(sub_x[ix * self.mert_processor.sampling_rate:min(
                (ix + 60) * self.mert_processor.sampling_rate, len(sub_x))],
                                              sampling_rate=self.mert_processor.sampling_rate,
                                              return_tensors="pt").to(self.device) for ix in
                          range(0, len(sub_x) // (self.mert_processor.sampling_rate * 60) + 1, 60)]
            aggoutputs = torch.zeros(1, 25, 1024).cpu()
            for inputs in all_inputs:
                with torch.no_grad():
                    outputs = self.mert_model(**inputs, output_hidden_states=True)
                all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
                sub_x = all_layer_hidden_states.mean(-2).unsqueeze(0).cpu()
                aggoutputs += sub_x
            aggoutputs /= len(all_inputs)
            sub_x = self.mu_mert_agg(aggoutputs).squeeze()
            xs.append(sub_x)
        x = torch.stack(xs, dim=0)
        return x

    def forward(self, inputs: List[str]) -> torch.Tensor:
        inputs = load_and_transform_audio_data(inputs)
        audio_feats = F.normalize(self.encode_audio(inputs), dim=-1)
        audio_feats = audio_feats.unsqueeze(1)

        audio_feats = self.mu_mert_proj(audio_feats)

        audio_feats_norm = self.mu_mert_norm_1(audio_feats)
        audio_feats = audio_feats + self.mu_mert_f2_1(
            F.silu(self.mu_mert_f1_1(audio_feats_norm)) * self.mu_mert_f3_1(audio_feats_norm))

        audio_feats_norm = self.mu_mert_norm_2(audio_feats)
        audio_feats = audio_feats + self.mu_mert_f2_2(
            F.silu(self.mu_mert_f1_2(audio_feats_norm)) * self.mu_mert_f3_2(audio_feats_norm))

        audio_feats_norm = self.mu_mert_norm_3(audio_feats)
        audio_feats = audio_feats + self.mu_mert_f2_3(
            F.silu(self.mu_mert_f1_3(audio_feats_norm)) * self.mu_mert_f3_3(audio_feats_norm))
        return audio_feats
