#!/bin/sh
#SBATCH -p edu-20h
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-5
module load cuda

conda run -n dl-venv python tpt_eval.py > out.txt