#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=0 python main.py --ds site-$1 > site-2/log.txt
#python main.py --ds site-$1 > site-$1/log.txt
workspace="workspace_medical_2018_challenge_baseline_$3"

rm -rf $workspace/site-$1
mkdir $workspace/
mkdir $workspace/site-$1
CUDA_VISIBLE_DEVICES=$2 python main.py \
    --ds site-$1 \
    --workspace $workspace \
    --model $3 > $workspace/site-$1/log.txt

# sbatch simulate_single.slurm 0 0 bert-base-uncased