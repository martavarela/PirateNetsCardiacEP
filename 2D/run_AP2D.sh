#!/bin/bash
#PBS -l select=1:ncpus=30:mem=250gb:ngpus=1
#PBS -l walltime=16:00:00
#PBS -m ae
#PBS -N 2D_planar_sphere
#PBS -o ./output/
#PBS -e ./output/

module load anaconda3/personal
source activate /rds/general/user/cc8418/home/anaconda3/envs/piratenets

cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
python /rds/general/user/cc8418/home/piratenets_cardiac_ep/test_2D/main_2D.py
conda deactivate