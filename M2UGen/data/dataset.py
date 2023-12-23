from .enc_datasets import *
from .dec_datasets import *
from .instruction_datasets import *
from torch.utils.data import ConcatDataset


class FinetuneDataset(Dataset):
    def __init__(self, max_words=30, tokenizer=None, stage=1):
        dataset_list = []

        if True:#stage == 1:
            # Encoder Datasets
            mucaps = MUCapsDataset("/hpctmp/e0589920/MUGen/data/MUCaps/MUCapsCaptions.json",
                                   "/hpctmp/e0589920/MUGen/data/MUCaps/audios/", "AudioToText", tokenizer, max_words)
            coco = COCODataset("/hpctmp/e0589920/MUGen/data/COCO/COCOCaptions.json",
                               "/hpctmp/e0589920/MUGen/data/COCO/train2014/", "ImageToText", tokenizer, max_words)
            videocaps = VideoCapsDataset("/hpctmp/e0589920/MUGen/data/MUVideo/MUVideoCaptions.json",
                                         "/hpctmp/e0589920/MUGen/data/MUVideo/audioset_video/", "VideoToText",
                                         tokenizer, max_words)
            dataset_list.append(mucaps)
            dataset_list.append(coco)
            dataset_list.append(videocaps)

        if True:#stage == 2:
            # Decoder Dataset
            mucaps_decoder = MUCapsDecoderDataset("/hpctmp/e0589920/MUGen/data/MUCaps/MUCapsCaptions.json",
                                                  "/hpctmp/e0589920/MUGen/data/MUCaps/audios/", "TextToAudio",
                                                  tokenizer, max_words)
            dataset_list.append(mucaps_decoder)

        if True:#stage == 3:
            # QA Dataset
            musicqa = MusicQADataset("/hpctmp/e0589920/MUGen/data/MusicQA/MusicQA.json",
                                     "/hpctmp/e0589920/MU-LLaMA/MusicQA/MusicQA/audios", "AudioToText", tokenizer,
                                     max_words)

            # Text Instruction
            alpaca = AlpacaDataset("/hpctmp/e0589920/MUGen/data/Alpaca/alpaca_data.json", "TextToText", tokenizer,
                                   max_words)

            # Generation Instruction Datasets
            muimage = AnyToMusicInstructionDataset("/hpctmp/e0589920/MUGen/data/MUImage/MUImageInstructions.json",
                                                   "/hpctmp/e0589920/MUGen/data/MUImage/audioset_images",
                                                   "/hpctmp/e0589920/MUGen/data/MUImage/audioset",
                                                   "ImageToAudio", tokenizer, max_words)
            muvideo = AnyToMusicInstructionDataset("/hpctmp/e0589920/MUGen/data/MUVideo/MUVideoInstructions.json",
                                                   "/hpctmp/e0589920/MUGen/data/MUVideo/audioset_video",
                                                   "/hpctmp/e0589920/MUGen/data/MUVideo/audioset",
                                                   "VideoToAudio", tokenizer, max_words)
            muedit = AnyToMusicInstructionDataset("/hpctmp/e0589920/MUGen/data/MUEdit/MUEditInstructions.json",
                                                  "/hpctmp/e0589920/MUGen/data/MUEdit/audioset",
                                                  "/hpctmp/e0589920/MUGen/data/MUEdit/audioset",
                                                  "AudioToAudio", tokenizer, max_words)
            dataset_list.append(musicqa)
            dataset_list.append(alpaca)
            dataset_list.append(muimage)
            dataset_list.append(muvideo)
            dataset_list.append(muedit)
        self.datasets = ConcatDataset(dataset_list)

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        return self.datasets[index]
