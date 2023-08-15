#!/bin/bash

source /data/vision/polina/projects/wmh/dhollidt/conda/bin/activate
conda activate nerfstudio2

# DATA_CONFIG="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_nesf_train_100.json"
# DATA_CONFIG="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/nesf_test_config_5.json"
# DATA_CONFIG="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_nesf_train_100.json"

ns-train nesf --load-config /data/vision/polina/projects/wmh/dhollidt/documents/nerf/nesf_models/tmp/nesf/2023-07-11_025311_244720/config.yml
