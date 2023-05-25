#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=40g
#SBATCH --tmp=40g
#SBATCH --mail-type=NONE  
#SBATCH --mail-user=peng0347@umn.edu 

##SBATCH -p v100                                   
##SBATCH --gres=gpu:v100:1

#SBATCH --gres=gpu:a100:1
#SBATCH -p a100-4
##SBATCH -p apollo_agate

module load parallel

cd /home/jusun/peng0347/FedNLP/NER
export PATH=/home/jusun/peng0347/miniconda3/env/torch2/bin:$PATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jusun/peng0347/miniconda3/lib/
sh ./bash_scripts/run.sh $1 $2 $3 $4 $5 $6 $7

# conda env create -f environment.yml
# sbatch simulate_single.slurm site-0 BC2GM bluebert 50
# ./bash_scripts/run.sh site-0 BC2GM bluebert 50

## models: bluebert, biobert, bio_clinicalbert, bert-base-uncased, gpt2, bi_lstm_crf

## dataset: 2018_n2c2, BC2GM, BC4CHEMD, BC5CDR-chem, BC5CDR-disease, JNLPBA, linnaeus, NCBI-disease, s800


## examples
# sbatch simulate_single.slurm site-0 feature_shift bi_lstm_crf 50 2 1 1
# sbatch simulate_single.slurm site-1 feature_shift bluebert 50 2 3 3
# sbatch simulate_single.slurm site-0 feature_shift biobert 50 2 3 3
# sbatch simulate_single.slurm site-0 feature_shift bio_clinicalbert 50 2 3 3
# sbatch simulate_single.slurm site-0 feature_shift bert-base-uncased 50 2 3 3
# sbatch simulate_single.slurm site-0 feature_shift gpt2 50 2 3 3