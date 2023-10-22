import os


class DatasetCatalog:
    def __init__(self):
        # the following dataset utilized for encoding-side alignment learning
        self.mucaps_enc = {
            "target": "dataset.encoder_dataset.MUCapsDataset",
            "params": dict(
                data_path="./data/MUCaps/MUCapsCaptions.json",
                mm_root_path="./data/MUCaps/audios/",
                embed_path="./data/MUCaps/embeds/",
                dataset_type="AudioToText",
            ),
        }

        self.coco_enc = {
            "target": "dataset.encoder_dataset.COCODataset",
            "params": dict(
                data_path="./data/COCO/COCOCaptions.json",
                mm_root_path="./data/COCO/train2014/",
                dataset_type="ImageToText",
            ),
        }

        self.videocaps_enc = {
            "target": "dataset.encoder_dataset.VideoCapsDataset",
            "params": dict(
                data_path="./data/MUVideo/MUVideoCaptions.json",
                mm_root_path="./data/MUVideo/audioset_video/",
                dataset_type="VideoToText",
            ),
        }

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

        # the following dataset utilized for decoding-side alignment learning.

        self.mucaps_dec = {
            "target": "dataset.encoder_dataset.MUCapsDataset",
            "params": dict(
                data_path="./data/MUCaps/MUCapsCaptions.json",
                mm_root_path="./data/MUCaps/audios/",
                embed_path="./data/MUCaps/embeds/",
                dataset_type="TextToAudio",
            ),
        }

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

        # the following dataset utilized for instruction tuning, so they are instruction dataset.

        self.musicqa_instruction = {
            "target": "dataset.T+X-T_instruction_dataset.TX2TInstructionDataset",
            "params": dict(
                data_path="./data/MusicQA/MusicQA.json",
                mm_root_path="../MU-LLaMA/MusicQA/MusicQA/audios",
                dataset_type="AudioToText",
            ),
        }

        self.muimage_instruction = {
            "target": "dataset.AnyToAny_instruction_dataset.AnyToAnyInstructionDataset",
            "params": dict(
                data_path="./data/MUImage/MUImageInstructions.json",
                input_root_path="./data/MUImage/audioset_images",
                output_root_path="./data/MUImage/audioset",
                embed_path="./data/MUImage/embeds",
                dataset_type="ImageToAudio",
            ),
        }

        self.muvideo_instruction = {
            "target": "dataset.AnyToAny_instruction_dataset.AnyToAnyInstructionDataset",
            "params": dict(
                data_path="./data/MUVideo/MUVideoInstructions.json",
                input_root_path="./data/MUVideo/audioset_video",
                output_root_path="./data/MUVideo/audioset",
                embed_path="./data/MUVideo/embeds",
                dataset_type="VideoToAudio",
            ),
        }

        self.muedit_instruction = {
            "target": "dataset.AnyToAny_instruction_dataset.AnyToAnyInstructionDataset",
            "params": dict(
                data_path="./data/MUEdit/MUEditInstructions.json",
                input_root_path="./data/MUEdit/audios",
                output_root_path="./data/MUEdit/audios",
                embed_path="./data/MUEdit/embeds",
                dataset_type="AudioToAudio",
            ),
        }

        self.alpaca_instruction = {
            "target": "dataset.T+X-T_instruction_dataset.TX2TInstructionDataset",
            "params": dict(
                data_path="./data/Alpaca/alpaca_data.json",
                dataset_type="TextToText",
            ),
        }
