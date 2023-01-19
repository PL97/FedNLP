#!/usr/bin/env bash


# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$2 python fed_main.py saved_dir $1
workspace="workspace_medical_2018_challenge_$3_split_test"
rm -rf $workspace/$1
mkdir $workspace/
mkdir $workspace/$1
CUDA_VISIBLE_DEVICES=$2 python fed_main.py \
    --workspace $workspace/$1 \
    --n_split $3 \
    > $workspace/$1/log.txt