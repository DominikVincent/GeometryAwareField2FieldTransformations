#!/bin/bash

set -x  # Enable debug mode to print each command


ns-train depth-nerfacto \
	--vis wandb \
	--pipeline.datamanager.camera-optimizer.mode off \
	--save-only-latest-checkpoint True \
	--pipeline.model.predict-normals True \
	--max-num-iterations 8000 \
	--data $1 \
	--experiment-name $2 \
	--output-dir $3 \
	--timestamp "07_04_23" \
	--viewer.websocket-port 7008
	# --optimizers.proposal-networks.optimizer.weight-decay 0.0001 \
