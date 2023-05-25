#!/usr/bin/env bash


# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$2 python fed_main.py saved_dir $1
workspace="workspace$7/$2/$4/$1"
rm -rf $workspace/$1 && mkdir -p $workspace || mkdir -p $workspace

CUDA_VISIBLE_DEVICES=0 python fed_main.py \
    --fedalg $1 \
    --ds $2 \
    --workspace $workspace/ \
    --n_split $3 \
    --model $4 \
    --seed $6 \
    --epochs $5 &> $workspace/log.txt

# ./bash_scripts/fed.sh fedavg BC2GM 10 bluebert 50
# sbatch simulate_fed.slurm fedavg BC2GM 10 bluebert 50