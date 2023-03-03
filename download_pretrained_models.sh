#!/usr/bin/env bash

git lfs install
mkdir pretrained_models && cd pretrained_models || cd pretrained_models

## BERT_base_uncased
git clone https://huggingface.co/bert-base-uncased

## BioBERT
git clone https://huggingface.co/dmis-lab/biobert-v1.1


## BlueBERT
readonly NCBI_DIR=pretrained_models/NCBI_BERT_pubmed_uncased_L-12_H-768_A-12/
git clone https://huggingface.co/bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16 && export NCBI_DIR=$1
transformers-cli convert --model_type bert \
  --tf_checkpoint $NCBI_DIR/bert_model.ckpt \
  --config $NCBI_DIR/bert_config.json \
  --pytorch_dump_output $NCBI_DIR/pytorch_model.bin

## Bio_ClinicalBERT
git clone https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT

## GPT2
git clone https://huggingface.co/gpt2


GIT_LFS_SKIP_SMUDGE=1


