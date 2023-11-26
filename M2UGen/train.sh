#!/usr/bin/bash

LLAMA_PATH="$1"
OUTPUT_DIR="$2"

mkdir -p "$OUTPUT_DIR"

# python3 -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=2 --use_env \
#  main_train.py --batch_size 1 --accum_iter 4 --stage 1 \
#  --epochs 2 --split_epoch 10 --warmup_epochs 2 --blr 1e-4 --weight_decay 0.05 \
#  --llama_path "$LLAMA_PATH" \
#  --output_dir "$OUTPUT_DIR"

# python3 -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=1 --use_env \
#  main_train.py --batch_size 1 --accum_iter 4 --stage 2 --min_lr 1e-4\
#  --epochs 5 --split_epoch 2 --warmup_epochs 1 --blr 1e-4 --weight_decay 0.05 \
#  --llama_path "$LLAMA_PATH" \
#  --output_dir "$OUTPUT_DIR"

 python3 -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=1 --use_env \
 main_train.py --batch_size 1 --accum_iter 4 --stage 3 --load_same_stage \
 --epochs 10000 --split_epoch 100 --warmup_epochs 0 --blr 1e-3 --min_lr 0 --weight_decay 0.05 \
 --llama_path "$LLAMA_PATH" \
 --output_dir "$OUTPUT_DIR"

#  python3 -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=1 --use_env \
#  main_train.py --batch_size 1 --accum_iter 2 --stage 3 --load_same_stage \
#  --epochs 50 --split_epoch 10 --warmup_epochs 2 --blr 1e-3 --weight_decay 0.05 \
#  --llama_path "$LLAMA_PATH" \
#  --output_dir "$OUTPUT_DIR"

#   python3 -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=1 --use_env \
#  main_train.py --batch_size 1 --accum_iter 2 --stage 3 --load_same_stage \
#  --epochs 50 --split_epoch 10 --warmup_epochs 2 --blr 1e-3 --weight_decay 0.05 \
#  --llama_path "$LLAMA_PATH" \
#  --output_dir "$OUTPUT_DIR"

#   python3 -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=1 --use_env \
#  main_train.py --batch_size 1 --accum_iter 2 --stage 3 --load_same_stage \
#  --epochs 50 --split_epoch 10 --warmup_epochs 2 --blr 1e-3 --weight_decay 0.05 \
#  --llama_path "$LLAMA_PATH" \
#  --output_dir "$OUTPUT_DIR"

#   python3 -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=1 --use_env \
#  main_train.py --batch_size 1 --accum_iter 2 --stage 3 --load_same_stage \
#  --epochs 50 --split_epoch 10 --warmup_epochs 2 --blr 1e-3 --weight_decay 0.05 \
#  --llama_path "$LLAMA_PATH" \
#  --output_dir "$OUTPUT_DIR"

#   python3 -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=1 --use_env \
#  main_train.py --batch_size 1 --accum_iter 2 --stage 3 --load_same_stage \
#  --epochs 50 --split_epoch 10 --warmup_epochs 2 --blr 1e-3 --weight_decay 0.05 \
#  --llama_path "$LLAMA_PATH" \
#  --output_dir "$OUTPUT_DIR"

#   python3 -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=1 --use_env \
#  main_train.py --batch_size 1 --accum_iter 2 --stage 3 --load_same_stage \
#  --epochs 50 --split_epoch 10 --warmup_epochs 2 --blr 1e-3 --weight_decay 0.05 \
#  --llama_path "$LLAMA_PATH" \
#  --output_dir "$OUTPUT_DIR"

#   python3 -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=1 --use_env \
#  main_train.py --batch_size 1 --accum_iter 2 --stage 3 --load_same_stage \
#  --epochs 50 --split_epoch 10 --warmup_epochs 2 --blr 1e-3 --weight_decay 0.05 \
#  --llama_path "$LLAMA_PATH" \
#  --output_dir "$OUTPUT_DIR"

#   python3 -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=1 --use_env \
#  main_train.py --batch_size 1 --accum_iter 2 --stage 3 --load_same_stage \
#  --epochs 50 --split_epoch 10 --warmup_epochs 2 --blr 1e-3 --weight_decay 0.05 \
#  --llama_path "$LLAMA_PATH" \
#  --output_dir "$OUTPUT_DIR"