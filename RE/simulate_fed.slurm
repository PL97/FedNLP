#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=40g
#SBATCH --tmp=40g
#SBATCH --mail-type=NONE  
#SBATCH --mail-user=peng0347@umn.edu 
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

##SBATCH -p v100                                   
##SBATCH --gres=gpu:v100:1

#SBATCH --gres=gpu:a100:1
#SBATCH -p a100-4
##SBATCH -p apollo_agate

module load parallel

cd /home/jusun/peng0347/FedNLP/RE
export PATH=/home/jusun/peng0347/miniconda3/bin:$PATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jusun/peng0347/miniconda3/lib/
sh ./bash_scripts/fed.sh $1 $2 $3 $4 $5 $6 $7


# conda env create -f environment.yml
# ./bash_scripts/fed.sh fedavg euadr 10 bluebert 50
# sbatch simulate_fed.slurm fedavg euadr 10 bluebert 50

## models: bluebert, biobert, bio_clinicalbert, bert-base-uncased, gpt2, bi_lstm_crf
## euadr

# sbatch simulate_fed.slurm fedavg euadr 10 bluebert 50 3 3
# sbatch simulate_fed.slurm fedavg euadr 10 biobert 50 3 3
# sbatch simulate_fed.slurm fedavg euadr 10 bio_clinicalbert 50 3 3
# sbatch simulate_fed.slurm fedavg euadr 10 bert-base-uncased 50 3 3
# sbatch simulate_fed.slurm fedavg euadr 10 gpt2 50 1 1
