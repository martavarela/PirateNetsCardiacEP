#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=30:mem=300gb:ngpus=1
#PBS -N 2D_planar_sphere
#PBS -o ./output/
#PBS -e ./output/

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate /rds/general/user/cc8418/home/anaconda3/envs/piratenets

cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
python /rds/general/user/cc8418/home/piratenets_cardiac_ep/test_2D/main_2D.py
conda deactivate