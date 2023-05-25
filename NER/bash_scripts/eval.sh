#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=0 python main.py --ds site-$1 > site-2/log.txt
#python main.py --ds site-$1 > site-$1/log.txt
# mkdir site-$1
workspace="workspace$4/$2/$3/baseline/"

# rm -rf $workspace/$1 && mkdir -p $workspace/$1 || mkdir -p $workspace/$1

TF_CPP_MIN_LOG_LEVEL="2" CUDA_VISIBLE_DEVICES=0 python main.py \
    --ds $2 \
    --split $1 \
    --workspace $workspace \
    --model $3 \
    --eval \
    &> $workspace/$1/metrics.txt

#./bash_scripts/eval.sh site-0 2018_n2c2 bi_lstm_crf 1

# dataset: 2018_n2c2, BC2GM, BC4CHEMD, BC5CDR-chem, BC5CDR-disease, JNLPBA, linnaeus, NCBI-disease, s800
