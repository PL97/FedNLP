#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=0 python validation.py \
#     --ds site-$2 \
#     --workspace $1\
#     --model $3

# # ./bash_scripts/eval_fed.sh workspace_medical_2018_challenge_10_split_bert-base-uncased/fedavg/global/ 0 bert-base-uncased
# # ./bash_scripts/eval_fed.sh workspace1/bert-base-uncased/fedavg/fedavg/global/ 0 BILSTM_CRF


workspace="workspace$5/$2/$4/$1/"
CUDA_VISIBLE_DEVICES=0 python fed_main.py \
    --fedalg $1 \
    --ds $2 \
    --workspace $workspace/ \
    --n_split $3 \
    --model $4 \
    --seed $5 \
    --test_file "test_200" \
    --eval &> $workspace/llm_exp_log.txt

# ./bash_scripts/eval_fed.sh fedavg 2018_n2c2 10 bluebert 1
# sbatch simulate_fed_eval.slurm fedavg BC2GM 10 bluebert 50