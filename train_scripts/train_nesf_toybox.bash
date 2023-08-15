#!/bin/bash

source /data/vision/polina/projects/wmh/dhollidt/conda/bin/activate
conda activate nerfstudio2

# DATA_CONFIG="/YOUR/PATH/HERE/data/klever_nesf_train_100.json"
# DATA_CONFIG="/YOUR/PATH/HERE/data/nesf_test_config_5.json"
# DATA_CONFIG="/YOUR/PATH/HERE/data/klever_depth_nesf_train_100.json"
DATA_CONFIG="/YOUR/PATH/HERE/data/toybox-5_nesf_train_100_270.0.json"


RAYS=16384
ns-train nesf --data /data/vision/polina/projects/wmh/dhollidt/datasets/klevr_nesf/0  \
	--output-dir /YOUR/PATH/HERE/nesf_models/ \
	--vis wandb \
	--machine.num-gpus 1 \
	--steps-per-eval-batch 100 \
    --steps-per-eval-image 500 \
    --steps-per-save 5000 \
    --max-num-iterations 5000000 \
	--pipeline.datamanager.steps-per-model 1 \
	--pipeline.datamanager.train-num-images-to-sample-from 4 \
	--pipeline.datamanager.train-num-times-to-repeat-images 4 \
	--pipeline.datamanager.eval-num-images-to-sample-from 4 \
	--pipeline.datamanager.eval-num-times-to-repeat-images 4 \
	--pipeline.datamanager.train-num-rays-per-batch $RAYS \
	--pipeline.datamanager.eval-num-rays-per-batch $RAYS \
	--pipeline.model.eval-num-rays-per-chunk $RAYS \
	--pipeline.model.sampler.surface-sampling True \
	--pipeline.model.sampler.samples-per-ray 10 \
	--pipeline.model.sampler.ground-points-count 500 \
	--pipeline.model.sampler.ground-tolerance 0.025 \
	--pipeline.model.batching-mode "sliced" \
	--pipeline.model.batch_size 1536 \
	--pipeline.model.mode semantics \
	--pipeline.model.pretrain False  \
	--pipeline.model.feature-generator-config.use-rgb True \
	--pipeline.model.feature-generator-config.use-dir-encoding True \
	--pipeline.model.feature-generator-config.use-pos-encoding True \
	--pipeline.model.feature-generator-config.use-density True \
	--pipeline.model.feature-generator-config.out-rgb-dim 16 \
	--pipeline.model.feature-generator-config.pos-encoder "sin" \
	--pipeline.model.feature-generator-config.out-density-dim 8 \
	--pipeline.model.feature-generator-config.use-normal-encoding True \
	--pipeline.model.feature-generator-config.rot-augmentation True \
	--pipeline.model.space-partitioning "random" \
	--pipeline.model.feature-transformer-model "custom" \
	--pipeline.model.feature-transformer-custom-config.num-layers 6 \
	--pipeline.model.feature-transformer-custom-config.num-heads 8 \
	--pipeline.model.feature-transformer-custom-config.dim-feed-forward 64 \
	--pipeline.model.feature-transformer-custom-config.dropout-rate 0.2 \
	--pipeline.model.feature-transformer-custom-config.feature-dim 64 \
	nesf-data \
	--data-config $DATA_CONFIG
