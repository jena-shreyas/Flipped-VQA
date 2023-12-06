#!/bin/bash
module purge
module load python/3.8
module load StdEnv/2020 gcc/9.3.0
module load opencv/4.8.0
source $HOME/envs/flipvqa/bin/activate
python3 -c "import cv2"
 
 