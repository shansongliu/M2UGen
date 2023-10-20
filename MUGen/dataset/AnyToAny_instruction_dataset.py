import json
import os.path

from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import re
import random
import numpy as np
import torch


# from .base_dataset import BaseDataset


class AnyToAnyInstructionDataset(Dataset):
    """
    T + X - T + X instruction Dataset
    """
    def __init__(self, data_path: str, input_root_path: str = None, output_root_path: str = None, embed_path : str = None, dataset_type: str='AnyToAny'):
        super(AnyToAnyInstructionDataset, self).__init__()

        self.input_root_path = input_root_path
        self.embed_path = embed_path
        self.instruction_list = []
        self.input_path_list = []
        self.output_path_list = []
        with open(data_path, 'r', encoding='utf-8') as f:
            res = json.load(f)
        for instance in tqdm(res, total=len(res)):
            self.instruction_list.append(instance['conversation'])
            self.input_path_list.append(os.path.join(input_root_path, instance['input_file']))
            self.output_path_list.append(os.path.join(output_root_path, instance['output_file']))
        self.dataset_type_list = [dataset_type for _ in range(len(self.instruction_list))]

    def __len__(self):  # number of instances
        return len(self.instruction_list)

    def __getitem__(self, i):
        with open(os.path.join(self.embed_path, str(os.path.basename(self.output_path_list[i])) + '.npy'), 'rb') as f:
            output_embs = torch.from_numpy(np.load(f, allow_pickle=True))
        return dict(input_paths=self.input_path_list[i], output_texts=self.instruction_list[i],
                    dataset_types=self.dataset_type_list[i], output_embs=output_embs)

    def collate(self, instances):
        input_paths, output_texts, dataset_types, output_embs = tuple(
            [instance[key] for instance in instances] for key in
            ("input_paths", "output_texts", "dataset_types", "output_embs"))
        return dict(
            input_paths=input_paths,
            output_texts=output_texts,
            dataset_types=dataset_types,
            output_embs=output_embs
        )
