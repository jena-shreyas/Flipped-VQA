#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=32G
# SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node 1
#SBATCH --time=5:0:0
# SBATCH --mem=16G
#SBATCH --account=def-egranger
#SBATCH --job-name=clip-vitl14
#SBATCH --output=clip_vitl14_%j.out
#SBATCH --error=clip_vitl14_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shreyas.jena.1@etsmtl.net

module purge
module load StdEnv/2020 gcc/9.3.0
module load opencv/4.8.0
source $HOME/envs/flipvqa/bin/activate
python -c "import cv2"

data_path=/scratch/jenas/BTP/Causal-VidQA/data/orig_data/dataset/
python extract_clip.py $data_path