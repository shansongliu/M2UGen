#!/usr/bin/bash

LLAMA_PATH="$1"
OUTPUT_DIR="$2"

mkdir -p "$OUTPUT_DIR"

 python3 -u -m torch.distributed.launch --master_port=1114 --nproc_per_node=1 --use_env \
 main_train.py --batch_size 1 --accum_iter 4 --stage 1 --max_lr 1e-4 --music_decoder musicgen \
 --epochs 2 --split_epoch 1 --warmup_epochs 0 --lr 1e-5--weight_decay 0.05 \
 --llama_path "$LLAMA_PATH" \
 --output_dir "$OUTPUT_DIR"

 python3 -u -m torch.distributed.launch --master_port=1114 --nproc_per_node=1 --use_env \
  main_train.py --batch_size 1 --accum_iter 4 --stage 2 --max_lr 1e-5 --music_decoder musicgen \
  --epochs 2 --split_epoch 1 --warmup_epochs 5 --lr 1e-5 --weight_decay 0.05 \
  --llama_path "$LLAMA_PATH" \
  --output_dir "$OUTPUT_DIR"

 python3 -u -m torch.distributed.launch --master_port=1114 --nproc_per_node=1 --use_env \
 main_train.py --batch_size 1 --accum_iter 4 --stage 3 --max_lr 1e-5 --music_decoder musicgen \
 --epochs 5 --split_epoch 1 --warmup_epochs 0 --lr 1e-5 --min_lr 0 --weight_decay 0.05 \
 --llama_path "$LLAMA_PATH" \
 --output_dir "$OUTPUT_DIR"
