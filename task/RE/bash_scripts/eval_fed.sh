#!/usr/bin/env bash

workspace="workspace_medical_2018_challenge"
CUDA_VISIBLE_DEVICES=0 python validation.py \
    --ds site-1 \
    --workspace $workspace/fedavg/site-1 \
    > $workspace/fedavg/site-1/metrics.txt

CUDA_VISIBLE_DEVICES=0 python validation.py \
    --ds site-2 \
    --workspace $workspace/fedavg/site-2 \
    > $workspace/fedavg/site-2/metrics.txt

CUDA_VISIBLE_DEVICES=0 python validation.py \
    --ds fedavg \
    --workspace $workspace/fedavg/global \
    > $workspace/fedavg/global/metrics.txt