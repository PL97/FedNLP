#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python validation.py \
    --ds site-$2 \
    --workspace $1\
    --model $3

# ./bash_scripts/eval_fed.sh workspace_medical_2018_challenge_10_split_bert-base-uncased/fedavg/global/ 0 bert-base-uncased
./bash_scripts/eval_fed.sh workspace_medical_2018_challenge_10_split_BILSTM_CRF/fedavg/global/ 0 BILSTM_CRF