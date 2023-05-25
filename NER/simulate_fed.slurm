#!/bin/bash -l
#SBATCH --time=56:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem=40g
#SBATCH --tmp=40g
#SBATCH --mail-type=NONE  
#SBATCH --mail-user=peng0347@umn.edu 

##SBATCH -p v100                                   
##SBATCH --gres=gpu:v100:1

#SBATCH --gres=gpu:a100:1
##SBATCH -p a100-4
#SBATCH -p apollo_agate

module load parallel

cd /home/jusun/peng0347/FedNLP/NER
export PATH=/home/jusun/peng0347/miniconda3/envs/torch2/bin:$PATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jusun/peng0347/miniconda3/lib/
sh ./bash_scripts/fed.sh $1 $2 $3 $4 $5 $6 $7 $8
#sh ./bash_scripts/fed.sh $1 $2 $3 $4 $5 3 3


#sh ./bash_scripts/fed.sh fedavg 2018_n2c2 10 bi_lstm_crf 50 2 2
#sh ./bash_scripts/fed.sh fedavg BC2GM 10 bi_lstm_crf 50 2 2
#sh ./bash_scripts/fed.sh fedavg BC4CHEMD 10 bi_lstm_crf 50 2 2
#sh ./bash_scripts/fed.sh fedavg JNLPBA 10 bi_lstm_crf 50 2 2
#sh ./bash_scripts/fed.sh fedavg NCBI-disease 10 bi_lstm_crf 50 2 2
#
#sh ./bash_scripts/fed.sh fedavg 2018_n2c2 10 bi_lstm_crf 50 3 3
#sh ./bash_scripts/fed.sh fedavg BC2GM 10 bi_lstm_crf 50 3 3
#sh ./bash_scripts/fed.sh fedavg BC4CHEMD 10 bi_lstm_crf 50 3 3
#sh ./bash_scripts/fed.sh fedavg JNLPBA 10 bi_lstm_crf 50 3 3
#sh ./bash_scripts/fed.sh fedavg NCBI-disease 10 bi_lstm_crf 50 3 3




# conda env create -f environment.yml
# ./bash_scripts/fed.sh fedavg BC2GM 10 bluebert 50
# sbatch simulate_fed.slurm fedavg BC2GM 10 bluebert 50


## models: bluebert, biobert, bio_clinicalbert, bert-base-uncased, gpt2, bi_lstm_crf

## dataset: 2018_n2c2, BC2GM, BC4CHEMD, BC5CDR-chem, BC5CDR-disease, JNLPBA, linnaeus, NCBI-disease, s800

## example
# sbatch simulate_fed.slurm fedprox 2018_n2c2 2 bluebert 20 1 1
# sbatch simulate_fed.slurm fedprox feature_shift 10 biobert 100 1 1 0.1
# sbatch simulate_fed.slurm fedprox 2018_n2c2 2 bio_clinicalbert 20 1 1
# sbatch simulate_fed.slurm fedprox 2018_n2c2 2 bert-base-uncased 20 1 1
# sbatch simulate_fed.slurm fedprox 2018_n2c2 2 gpt2 20 1 1
# sbatch simulate_fed.slurm fedprox 2018_n2c2 2 bi_lstm_crf 20 1 1
