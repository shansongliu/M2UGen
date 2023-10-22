import os
import json
from .base_dataset import BaseDataset
from tqdm import tqdm
from .utils import process_caption


class MUCapsDataset(BaseDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, mm_root_path: str, embed_path: str, dataset_type: str):
        super(MUCapsDataset, self).__init__(data_path, mm_root_path, embed_path, dataset_type)
        self.embed_path = embed_path

        print('Load MUCaps dataset ...')
        self.mm_path_list, self.caption_list = [], []
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for audio_id, one_caption in tqdm(data.items(), total=len(data)):
            self.mm_path_list.append(os.path.join(mm_root_path, audio_id))
            self.caption_list.append(process_caption(one_caption))

        print(f'[!] collect {len(self.mm_path_list)} samples for training')
        self.dataset_type_list = [dataset_type for _ in range(len(self.caption_list))]


class VideoCapsDataset(BaseDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, mm_root_path: str, dataset_type: str):
        super(VideoCapsDataset, self).__init__(data_path, mm_root_path, "", dataset_type)

        print('Load VideoCaps dataset ...')
        self.mm_path_list, self.caption_list = [], []
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for video_id, one_caption in tqdm(data.items(), total=len(data)):
            self.mm_path_list.append(os.path.join(mm_root_path, video_id))
            self.caption_list.append(process_caption(one_caption))

        print(f'[!] collect {len(self.mm_path_list)} samples for training')
        self.dataset_type_list = [dataset_type for _ in range(len(self.caption_list))]


class ImageCapsDataset(BaseDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, mm_root_path: str, dataset_type: str):
        super(ImageCapsDataset, self).__init__(data_path, mm_root_path, "", dataset_type)

        print('Load ImageCaps dataset ...')
        self.mm_path_list, self.caption_list = [], []
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for video_id, one_caption in tqdm(data.items(), total=len(data)):
            self.mm_path_list.append(os.path.join(mm_root_path, video_id))
            self.caption_list.append(process_caption(one_caption))

        print(f'[!] collect {len(self.mm_path_list)} samples for training')
        self.dataset_type_list = [dataset_type for _ in range(len(self.caption_list))]


class COCODataset(BaseDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, mm_root_path: str, dataset_type: str):
        super(COCODataset, self).__init__(data_path, mm_root_path, "", dataset_type)

        print('Load COCO dataset ...')
        self.mm_path_list, self.caption_list = [], []
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for video_id, one_caption in tqdm(data.items(), total=len(data)):
            self.mm_path_list.append(os.path.join(mm_root_path, video_id))
            self.caption_list.append(process_caption(one_caption))

        print(f'[!] collect {len(self.mm_path_list)} samples for training')
        self.dataset_type_list = [dataset_type for _ in range(len(self.caption_list))]