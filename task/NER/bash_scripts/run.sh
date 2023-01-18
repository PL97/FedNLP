#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=0 python main.py --ds site-$1 > site-2/log.txt
#python main.py --ds site-$1 > site-$1/log.txt
workspace="workspace_medical_2018_challenge_10_split"

rm -rf $workspace/site-$1
mkdir $workspace/
mkdir $workspace/site-$1
CUDA_VISIBLE_DEVICES=$2 python main.py \
    --ds site-$1 \
    --workspace $workspace \
    > $workspace/site-$1/log.txt
