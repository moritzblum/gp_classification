#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-7:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c10
#SBATCH -o /homes/mblum/outputs/stdout_gp_%j_%t

source /homes/mblum/.bashrc
source /homes/mblum/miniconda3/etc/profile.d/conda.sh

conda activate gp

cd /homes/mblum/gp_classification
echo "start GP evaluation"

# GCN models
python main.py --model GCN --freeze_emb --run_name test
python main.py --model GCN --run_name test

# R-GCN models
python main.py --model RGCN --freeze_emb --run_name test
python main.py --model RGCN --run_name test

# GINE models
python main.py --model GINE --freeze_emb --run_name test
python main.py --model GINE --run_name test
