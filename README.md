## FedNLP

Git repo for paper "Federated Learning for Clinical Natural Language Processing"

## Overview
| task      | models | datasets| FL |
| ----------- | ----------- |----------|----------|
| NER      | BERT-base; BlueBERT; BioBERT; GPT2; BiLSTM-CRF       |2018_n2c2; BC2GM; BC5CDR-disease; JNLPBA; NCBI-disease [download :link:](https://drive.google.com/drive/folders/1m7q3f3oVCtyAGn8L540l6AKSPx2UF9wk?usp=share_link)  | FedAvg |
| RE   | BERT-base; BlueBERT; BioBERT; GPT2  | 2018_n2c2; euadr; GAD [download :link:](https://drive.google.com/drive/folders/1xdRDaT_RxIopIPNgNh7Y2Nkwj7UHK-G5?usp=sharing)| FedAvg |

## Installation
```bash
git clone https://github.com/PL97/FedNLP.git
cd FedNLP/NER

pip install -r requirements.txt
```

## Datasets
Download the dataset using the link in the table. Rename ***RE***/***NER*** to ***data*** and place under ***FedNLP/NER/*** and ***FedNLP/RE/*** respectively.


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
@misc{FedNLP,
    author={Le Peng},
    title={FedNLP: Federated Learning for Clinical Natural Language Processing},
    howpublished={\url{https://github.com/PL97/FedNLP.git}},
    year={2023}
}
```
