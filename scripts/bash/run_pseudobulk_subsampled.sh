#!/bin/bash
#SBATCH -o pbulk_sub_%j.out
#SBATCH -e pbulk_sub_%j.err
#SBATCH -J pbulk_sub
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=80G
#SBATCH -t 0-01:00:00
#SBATCH --nice=1

source /home/haicu/soeren.becker/.bashrc
conda activate /home/haicu/soeren.becker/miniconda3/envs/env_sc_dask2

echo "Starting."

python /home/haicu/soeren.becker/repos/HumanCytokineDict/scripts/compute_pseudobulks_subsampled.py \
    --multirun \
    dataset=humanCytokineDict \
    +hparams=pseudobulks_subsampled_per_condition
