#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100l:2
#SBATCH --ntasks-per-node 1
#SBATCH --time=12:0:0
#SBATCH --account=def-egranger
#SBATCH --job-name=flipvqa-cvqa
#SBATCH --output=flipvqa_cvqa_%j.out
#SBATCH --error=flipvqa_cvqa_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shreyas.jena.1@etsmtl.net

module purge
module load python/3.8
module load StdEnv/2020 gcc/9.3.0
module load opencv/4.8.0
source $HOME/envs/flipvqa/bin/activate
python3 -c "import cv2"
 
cd $SCRATCH/BTP/Flipped-VQA
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node 1 train.py --model 7B --max_seq_len 256 --batch_size 2 --epochs 5 --warmup_epochs 2 --bias 3.5 --tau 100. --max_feats 10 --dataset causalvidqa --blr 9e-2 --weight_decay 0.14 --output_dir ./checkpoint/causalvidqa --accum_iter 1 --num_workers 0 --no_pin_mem --vaq --qav
