#!/usr/bin/env bash

./bash_scripts/eval.sh site-0 2018_n2c2 bluebert
./bash_scripts/eval.sh site-0 2018_n2c2 biobert
./bash_scripts/eval.sh site-0 2018_n2c2 bio_clinicalbert
./bash_scripts/eval.sh site-0 2018_n2c2 bert-base-uncased

./bash_scripts/eval.sh site-1 2018_n2c2 bluebert
./bash_scripts/eval.sh site-1 2018_n2c2 biobert
./bash_scripts/eval.sh site-1 2018_n2c2 bio_clinicalbert
./bash_scripts/eval.sh site-1 2018_n2c2 bert-base-uncased


./bash_scripts/eval_fed.sh fedavg 2018_n2c2 10 bluebert
./bash_scripts/eval_fed.sh fedavg 2018_n2c2 10 biobert
./bash_scripts/eval_fed.sh fedavg 2018_n2c2 10 bio_clinicalbert
./bash_scripts/eval_fed.sh fedavg 2018_n2c2 10 bert-base-uncased


## other clients
./bash_scripts/eval.sh site-2 2018_n2c2 bluebert
./bash_scripts/eval.sh site-2 2018_n2c2 biobert
./bash_scripts/eval.sh site-2 2018_n2c2 bio_clinicalbert
./bash_scripts/eval.sh site-2 2018_n2c2 bert-base-uncased


./bash_scripts/eval.sh site-3 2018_n2c2 bluebert
./bash_scripts/eval.sh site-3 2018_n2c2 biobert
./bash_scripts/eval.sh site-3 2018_n2c2 bio_clinicalbert
./bash_scripts/eval.sh site-3 2018_n2c2 bert-base-uncased



./bash_scripts/eval.sh site-4 2018_n2c2 bluebert
./bash_scripts/eval.sh site-4 2018_n2c2 biobert
./bash_scripts/eval.sh site-4 2018_n2c2 bio_clinicalbert
./bash_scripts/eval.sh site-4 2018_n2c2 bert-base-uncased



./bash_scripts/eval.sh site-5 2018_n2c2 bluebert
./bash_scripts/eval.sh site-5 2018_n2c2 biobert
./bash_scripts/eval.sh site-5 2018_n2c2 bio_clinicalbert
./bash_scripts/eval.sh site-5 2018_n2c2 bert-base-uncased



./bash_scripts/eval.sh site-6 2018_n2c2 bluebert
./bash_scripts/eval.sh site-6 2018_n2c2 biobert
./bash_scripts/eval.sh site-6 2018_n2c2 bio_clinicalbert
./bash_scripts/eval.sh site-6 2018_n2c2 bert-base-uncased



./bash_scripts/eval.sh site-7 2018_n2c2 bluebert
./bash_scripts/eval.sh site-7 2018_n2c2 biobert
./bash_scripts/eval.sh site-7 2018_n2c2 bio_clinicalbert
./bash_scripts/eval.sh site-7 2018_n2c2 bert-base-uncased



./bash_scripts/eval.sh site-8 2018_n2c2 bluebert
./bash_scripts/eval.sh site-8 2018_n2c2 biobert
./bash_scripts/eval.sh site-8 2018_n2c2 bio_clinicalbert
./bash_scripts/eval.sh site-8 2018_n2c2 bert-base-uncased



./bash_scripts/eval.sh site-9 2018_n2c2 bluebert
./bash_scripts/eval.sh site-9 2018_n2c2 biobert
./bash_scripts/eval.sh site-9 2018_n2c2 bio_clinicalbert
./bash_scripts/eval.sh site-9 2018_n2c2 bert-base-uncased


./bash_scripts/eval.sh site-10 2018_n2c2 bluebert
./bash_scripts/eval.sh site-10 2018_n2c2 biobert
./bash_scripts/eval.sh site-10 2018_n2c2 bio_clinicalbert
./bash_scripts/eval.sh site-10 2018_n2c2 bert-base-uncased

