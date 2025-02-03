#!/bin/bash
#PBS -l select=1:ncpus=4:mem=250gb:ngpus=1
#PBS -l walltime=8:00:00
#PBS -m ae
#PBS -N 3D_planar_sphere
#PBS -o ./output/
#PBS -e ./output/

module load anaconda3/personal
source activate /rds/general/user/cc8418/home/anaconda3/envs/piratenets

cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
python main.py
conda deactivate