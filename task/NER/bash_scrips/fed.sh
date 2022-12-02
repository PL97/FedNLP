#!/usr/bin/env bash


# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$2 python fed_main.py saved_dir $1
CUDA_VISIBLE_DEVICES=$2 python fed_main.py \
    --saved_dir workspace/$1