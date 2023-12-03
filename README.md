<p>
  <h1>
    <img src="./assets/logo.png" height=120px align="right"/>
    M<sup>2</sup>UGen: Multi-modal Music Understanding and Generation<br/> with the Power of Large Language Models
  </h1>
</p>

[![PWC](https://img.shields.io/badge/%F0%9F%93%8E%20arXiv-Paper-red)](https://arxiv.org/abs/2311.11255)

This is the official repository for *[M<sup>2</sup>UGen: Multi-modal Music Understanding and Generation with the Power of Large Language Models](https://arxiv.org/abs/2308.11276)*.

## Introduction

The M<sup>2</sup>UGen model is a Music Understanding and Generation model that is capable of Music Question Answering and also Music Generation from texts, images, videos and audios. The model utilizes encoders such as MERT for music understanding, ViT for image understanding and ViViT for video understanding and the MusicGen/AudioLDM2 model as the music generation model, coupled with adapters and the LLaMA model to make the model possible. The model architecture is given in [**_m2ugen.py_**](./M2UGen/llama/m2ugen.py). 

<p align="center">
  <img src="./assets/M2UGen.png">
</p>

To train our model, we generate datasets using the [MU-LLaMA](https://github.com/crypto-code/MU-LLaMA) model. The dataset generation methods are given in the [Datasets](./Datasets) folder. 

## Model Training

To train the M<sup>2</sup>UGen model, run the [**_train.sh_**](./train.sh) script. The script is designed to run traiing for all three stages.

The main model architecture is given in [**_m2ugen.py_**](./llama/m2ugen.py) and the modified MusicGen architecture is present within the [**_musicgen_**](./llama/musicgen/) folder. The [**_data_**](./data/) folder contains the python files to handle loading and iterating through the dataset. The [**_data.py_**](./data/dataset.py) file will show the use of different datasets based on the training stage. The code for the training epochs are present in [**_engine_train.py_**](./engine_train.py).

## Model Testing

To test the M<sup>2</sup>UGen model, run gradio_app.py.

```
usage: gradio_app.py [-h] [--model MODEL] [--llama_type LLAMA_TYPE] [--llama_dir LLAMA_DIR] [--mert_path MERT_PATH] [--vit_path VIT_PATH] [--vivit_path VIVIT_PATH] [--knn_dir KNN_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Name of or path to M2UGen pretrained checkpoint
  --llama_type LLAMA_TYPE
                        Type of llama original weight
  --llama_dir LLAMA_DIR
                        Path to LLaMA pretrained checkpoint
  --mert_path MERT_PATH
                        Path to MERT pretrained checkpoint
  --vit_path VIT_PATH   Path to ViT pretrained checkpoint
  --vivit_path VIVIT_PATH
                        Path to ViViT pretrained checkpoint
  --knn_dir KNN_DIR     Path to directory with KNN Index
```
