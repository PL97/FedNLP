#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=0 python main.py --ds site-$1 > site-2/log.txt
#python main.py --ds site-$1 > site-$1/log.txt
# mkdir site-$1
workspace="workspace_medical_2018_challenge_baseline_$3"
CUDA_VISIBLE_DEVICES=$2 python validation.py \
    --ds site-$1 \
    --workspace $workspace/site-$1 \
    > $workspace/site-$1/metrics.txt

#./bash_scripts/eval.sh 0 0 bluebert
