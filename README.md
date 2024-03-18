## FedNLP

Offical git repo of paper "An In-Depth Evaluation of Federated Learning on Biomedical Natural Language Processing for Information Extraction"

## First stable release

### What we have :star2: 

- [x] support models with various architectures including BERT, GPT, and BILSTM-CRF
- [x] simulate federated learning using FedAvg
- [x] log the result using tensorboard
- [x] auxiliary bash script to download models from hugging face, run batch python script for fast prototype
- [x] simulate distribution shift in federated learning
- [x] study the impact of federated learning under different federation scales
- [x] study the impact of different federated learning algorithms

### What is expected to see in the next release :rocket: 
- [ ] comparison with LLM



## Overview
| task      | models | datasets| FL |
| ----------- | ----------- |----------|----------|
| NER      | BERT-base-uncased; BlueBERT; BioBERT; Bio_ClinicalBERT; GPT2; BiLSTM-CRF       |2018_n2c2; BC2GM; BC5CDR-disease; JNLPBA; NCBI-disease [download :link:](https://drive.google.com/drive/folders/1m7q3f3oVCtyAGn8L540l6AKSPx2UF9wk?usp=share_link)  | FedAvg |
| RE   | BERT-base-uncased; BlueBERT; BioBERT; Bio_ClinicalBERT; GPT2  | 2018_n2c2; euadr; GAD [download :link:](https://drive.google.com/drive/folders/1xdRDaT_RxIopIPNgNh7Y2Nkwj7UHK-G5?usp=sharing)| FedAvg |

## Installation
```bash
git clone https://github.com/PL97/FedNLP.git
cd FedNLP/

## setup running environments
conda create -n fednlp python==3.9.12
conda activate fednlp
pip install -r requirements.txt

## download model from hugging face
chmod +x download_pretrained_models.sh
./download_pretrained_models.sh
```

## Datasets
Download the dataset using the link in the table. Rename ***NER***/***RE*** to ***data*** and place under ***FedNLP/NER/*** and ***FedNLP/RE/*** respectively.


## Usage
### Named Entity Recognition (NER)

**centralized training** :point_down:

```bash
cd FedNLP/NER
chmod -R +x bash_scripts/

## run from python script
mkdir -p workspace_BC2GM/bluebert/baseline/
python main.py \
    --ds BC2GM \
    --split site-0 \
    --workspace  workspace_BC2GM/bluebert/baseline/\
    --model bluebert \
    --epochs 50


## alternatively can run from bash script (recommended)
./bash_scripts/run.sh site-0 BC2GM bluebert 50  ## arg1: data split; arg2: dataset; arg3: model; arg4: total epochs
```

**federated training** :point_down:

```bash
cd FedNLP/NER

## run from bash script (recommended)
 ./bash_scripts/fed.sh fedavg BC2GM 10 bluebert 50  ## arg1: FL algorithm; arg2: dataset; arg3: total data splits; arg4: model; arg5: total epochs

```

### Relation Extraction (RE)
**centralized training** :point_down:
```bash
cd FedNLP/RE

chmod -R +x bash_scripts/
./bash_scripts/run.sh site-0 euadr bluebert 50 ## arg1: data split; arg2: datasets; arg3: model; arg4: total epochs
```
**federated learning** :point_down:
```bash
cd FedNLP/RE

 ./bash_scripts/fed.sh fedavg euadr 10 bluebert 50  ## arg1: FL algorithm; arg2: dataset; arg3: total data splits; arg4: model; arg5: total epochs
```

## How to cite this work

```bibtex
@inproceedings{
    peng2023a,
    title={A Systematic Evaluation of Federated Learning on Biomedical Natural Language Processing},
    author={Le Peng and sicheng zhou and jiandong chen and Rui Zhang and Ziyue Xu and Ju Sun},
    booktitle={International Workshop on Federated Learning for Distributed Data Mining},
    year={2023},
    url={https://openreview.net/forum?id=pLEQFXACNA}
}
```
