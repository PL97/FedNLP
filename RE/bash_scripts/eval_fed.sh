#!/usr/bin/env bash

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$2 python fed_main.py saved_dir $1
workspace="workspace_$2/$4/$1/"

CUDA_VISIBLE_DEVICES=0 python fed_main.py \
    --ds $2 \
    --workspace $workspace/ \
    --n_split $3 \
    --model $4 \
    --eval > $workspace/metric.txt

# ./bash_scripts/eval_fed.sh fedavg euadr 10 bluebert
# sbatch simulate_fed.slurm fedavg euadr 10 bluebert 50