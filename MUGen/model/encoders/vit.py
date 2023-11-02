import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import List

from transformers import ViTImageProcessor, ViTModel


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


def load_and_transform_image_data(image_paths):
    images = []
    for path in image_paths:
        images.append(Image.open(path))
    return images


class ViTEncoder(nn.Module):

    def __init__(self, vit_path="/hpctmp/e0589920/ViT"):
        super().__init__()
        print(f'Initializing ViT encoder from {vit_path} ...')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vit_model = ViTModel.from_pretrained(vit_path).to(self.device)
        self.vit_model.eval()
        self.vit_processor = ViTImageProcessor.from_pretrained(vit_path)
        self.iu_vit_agg = nn.Conv1d(in_channels=197, out_channels=1, kernel_size=1)
        self.iu_vit_proj = nn.Linear(768, 4096)

        bridge_norm_layer = RMSNorm
        bridge_bias = False

        self.iu_vit_norm_1 = bridge_norm_layer(4096)
        self.iu_vit_f1_1 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)
        self.iu_vit_f2_1 = nn.Linear(4096 * 4, 4096, bias=bridge_bias)
        self.iu_vit_f3_1 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)

        self.iu_vit_norm_2 = bridge_norm_layer(4096)
        self.iu_vit_f1_2 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)
        self.iu_vit_f2_2 = nn.Linear(4096 * 4, 4096, bias=bridge_bias)
        self.iu_vit_f3_2 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)

        self.iu_vit_norm_3 = bridge_norm_layer(4096)
        self.iu_vit_f1_3 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)
        self.iu_vit_f2_3 = nn.Linear(4096 * 4, 4096, bias=bridge_bias)
        self.iu_vit_f3_3 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)
        print('ViT encoder initialized.')

    def encode_image(self, x) -> torch.Tensor:
        xs = []
        for sub_x in x:
            inputs = self.vit_processor(images=sub_x, return_tensors="pt").to(self.vit_model.device)
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            sub_x = self.iu_vit_agg(last_hidden_states).squeeze()
            xs.append(sub_x)
        return torch.stack(xs, dim=0)

    def forward(self, inputs: List[str]) -> torch.Tensor:
        inputs = load_and_transform_image_data(inputs)
        image_feats = F.normalize(self.encode_image(inputs), dim=-1)
        image_feats = image_feats.unsqueeze(1)

        image_feats = self.iu_vit_proj(image_feats)

        image_feats_norm = self.iu_vit_norm_1(image_feats)
        image_feats = image_feats + self.iu_vit_f2_1(
            F.silu(self.iu_vit_f1_1(image_feats_norm)) * self.iu_vit_f3_1(image_feats_norm))

        image_feats_norm = self.iu_vit_norm_2(image_feats)
        image_feats = image_feats + self.iu_vit_f2_2(
            F.silu(self.iu_vit_f1_2(image_feats_norm)) * self.iu_vit_f3_2(image_feats_norm))

        image_feats_norm = self.iu_vit_norm_3(image_feats)
        image_feats = image_feats + self.iu_vit_f2_3(
            F.silu(self.iu_vit_f1_3(image_feats_norm)) * self.iu_vit_f3_3(image_feats_norm))
        return image_feats