#!/bin/bash

#deepspeed --include localhost:0 --master_addr 127.0.0.1 --master_port 28459

python3 train.py \
    --model mugen \
    --stage 1\
    --batch_size 8\
    --save_path  ./ckpt/MUGen/7b_llama_v2\
    --log_path ./ckpt/MUGen/7b_llama_v2/log

