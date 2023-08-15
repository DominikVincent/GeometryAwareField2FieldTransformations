#!/bin/bash

source /data/vision/polina/projects/wmh/dhollidt/conda/bin/activate
conda activate nerfstudio3

python /data/vision/polina/projects/wmh/dhollidt/documents/nerf/nerfstudio_fork/scripts/sweep_nesf.py $1
