#!/bin/bash

set -x  # Enable debug mode to print each command

source /data/vision/polina/projects/wmh/dhollidt/conda/bin/activate

conda activate nerfstudio2

ns-train depth-nerfacto \
	--vis wandb \
	--pipeline.datamanager.camera-optimizer.mode off \
	--save-only-latest-checkpoint True \
	--pipeline.model.predict-normals True \
	--max-num-iterations 10000 \
	--data $1 \
	--experiment-name $2 \
	--output-dir $3 \
	--pipeline.model.pred-normal-loss-mult 0.01 \
	--timestamp "07_04_23" \
    --pipeline.model.is-euclidean-depth True \
    --pipeline.model.max-res 1536 \
    --pipeline.model.depth-sigma 0.001 \
    --pipeline.model.depth-loss-mult 0.01 \
	--viewer.websocket-port 7006  \
	--wandb-project-name "nesf-models-project"
	# --optimizers.proposal-networks.optimizer.weight-decay 0.0001 \
