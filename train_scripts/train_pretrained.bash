#!/bin/bash

source /data/vision/polina/projects/wmh/dhollidt/conda/bin/activate
conda activate nerfstudio2

ns-train nesf --load_config /YOUR/PATH/HERE/nesf_models/tmp/nesf/2023-05-03_205522/auto_semantic_config.yml