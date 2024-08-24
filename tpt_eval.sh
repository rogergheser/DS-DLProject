#!/bin/sh
#SBATCH -p edu-20h
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-20
module load cuda

conda run -n dl-venv python tpt_eval.py --run_name "imagenetV2/baseline-p1" > out.txt
rm -rf data/imagenetv2-random-partition/

conda run -n dl-venv python tpt_eval.py --run_name "imagenetV2/baseline-p2" > out.txt
rm -rf data/imagenetv2-random-partition/

conda run -n dl-venv python tpt_eval.py --run_name "imagenetV2/baseline-p3" > out.txt
rm -rf data/imagenetv2-random-partition/

conda run -n dl-venv python tpt_eval.py --run_name "imagenetV2/baseline-p4" > out.txt
rm -rf data/imagenetv2-random-partition/

conda run -n dl-venv python tpt_eval.py --run_name "imagenetV2/baseline-p5" > out.txt
rm -rf data/imagenetv2-random-partition/

