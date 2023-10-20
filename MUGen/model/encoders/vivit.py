import torch
import torch.nn as nn
import torch.nn.functional as F
import av
import numpy as np
from typing import List

from transformers import VivitImageProcessor, VivitModel


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


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def load_and_transform_video_data(image_paths):
    videos = []
    for path in image_paths:
        container = av.open(path)
        indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container=container, indices=indices)
        videos.append(video)
    return videos


class ViViTEncoder(nn.Module):

    def __init__(self, vivit_path="/hpctmp/e0589920/ViViT"):
        super().__init__()
        print(f'Initializing ViViT encoder from {vivit_path} ...')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vivit_model = VivitModel.from_pretrained(vivit_path)
        self.vivit_model.eval()
        self.vivit_processor = VivitImageProcessor.from_pretrained(vivit_path)
        self.iu_vivit_agg = nn.Conv1d(in_channels=3137, out_channels=1, kernel_size=1)
        self.iu_vivit_proj = nn.Linear(768, 4096)

        bridge_norm_layer = RMSNorm
        bridge_bias = False

        self.iu_vivit_norm_1 = bridge_norm_layer(4096)
        self.iu_vivit_f1_1 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)
        self.iu_vivit_f2_1 = nn.Linear(4096 * 4, 4096, bias=bridge_bias)
        self.iu_vivit_f3_1 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)

        self.iu_vivit_norm_2 = bridge_norm_layer(4096)
        self.iu_vivit_f1_2 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)
        self.iu_vivit_f2_2 = nn.Linear(4096 * 4, 4096, bias=bridge_bias)
        self.iu_vivit_f3_2 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)

        self.iu_vivit_norm_3 = bridge_norm_layer(4096)
        self.iu_vivit_f1_3 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)
        self.iu_vivit_f2_3 = nn.Linear(4096 * 4, 4096, bias=bridge_bias)
        self.iu_vivit_f3_3 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)
        print('ViViT encoder initialized.')

    def encode_video(self, x) -> torch.Tensor:
        xs = []
        for sub_x in x:
            inputs = self.vivit_processor(list(sub_x), return_tensors="pt")
            with torch.no_grad():
                outputs = self.vivit_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            sub_x = self.iu_vivit_agg(last_hidden_states).squeeze()
            xs.append(sub_x)
        return torch.stack(xs, dim=0)

    def forward(self, inputs: List[str]) -> torch.Tensor:
        inputs = load_and_transform_video_data(inputs)
        video_feats = F.normalize(self.encode_video(inputs), dim=-1)
        video_feats = video_feats.unsqueeze(1)

        video_feats = self.iu_vivit_proj(video_feats)

        video_feats_norm = self.iu_vivit_norm_1(video_feats)
        video_feats = video_feats + self.iu_vivit_f2_1(
            F.silu(self.iu_vivit_f1_1(video_feats_norm)) * self.iu_vivit_f3_1(video_feats_norm))

        video_feats_norm = self.iu_vivit_norm_2(video_feats)
        video_feats = video_feats + self.iu_vivit_f2_2(
            F.silu(self.iu_vivit_f1_2(video_feats_norm)) * self.iu_vivit_f3_2(video_feats_norm))

        video_feats_norm = self.iu_vivit_norm_3(video_feats)
        video_feats = video_feats + self.iu_vivit_f2_3(
            F.silu(self.iu_vivit_f1_3(video_feats_norm)) * self.iu_vivit_f3_3(video_feats_norm))
        return video_feats
