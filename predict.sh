#!/bin/bash
#$ -t 1-100
#$ -N Lorenz_predict_L4
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -m beas
#$ -o ./output
#$ -e ./error
#$ -q batch.q
python predict_anneal_steps.py 300 $SGE_TASK_ID
