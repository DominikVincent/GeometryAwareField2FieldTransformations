#!/bin/bash

source /data/vision/polina/projects/wmh/dhollidt/conda/bin/activate
conda activate nerfstudio2

# DATA_CONFIG="/YOUR/PATH/HERE/data/klever_nesf_train_100.json"
# DATA_CONFIG="/YOUR/PATH/HERE/data/klever_depth_nesf_train_10.json"
DATA_CONFIG="/YOUR/PATH/HERE/data/klever_depth_nesf_train_100.json"
# DATA_CONFIG="/YOUR/PATH/HERE/data/toybox-5_nesf_train_100_270.json"

RAYS=131072
# RAYS=65536
# RAYS=32768
# RAYS=16384
# EVAL_RAYS=65536
EVAL_RAYS=131072
ns-train nesf --data /data/vision/polina/projects/wmh/dhollidt/datasets/klevr_nesf/0  \
	--output-dir /YOUR/PATH/HERE/nesf_models/ \
	--vis wandb \
	--machine.num-gpus 1 \
	--steps-per-eval-batch 100 \
    --steps-per-eval-image 500 \
    --steps-per-save 5000 \
    --max-num-iterations 5000000 \
	--pipeline.datamanager.steps-per-model 1 \
	--pipeline.datamanager.train-num-images-to-sample-from 8 \
	--pipeline.datamanager.train-num-times-to-repeat-images 4 \
	--pipeline.datamanager.eval-num-images-to-sample-from 8 \
	--pipeline.datamanager.eval-num-times-to-repeat-images 4 \
	--pipeline.datamanager.train-num-rays-per-batch $RAYS \
	--pipeline.datamanager.eval-num-rays-per-batch $EVAL_RAYS \
    --pipeline.datamanager.use-sample-mask False \
    --pipeline.datamanager.sample-mask-ground-percentage 1.0 \
	--pipeline.model.eval-num-rays-per-chunk $EVAL_RAYS \
	--pipeline.model.sampler.surface-sampling True \
	--pipeline.model.sampler.samples-per-ray 24 \
	--pipeline.model.sampler.get-normals False \
	--pipeline.model.sampler.ground_removal_mode "ransac" \
	--pipeline.model.sampler.ground-points-count 500 \
	--pipeline.model.sampler.ground-tolerance 0.01 \
	--pipeline.model.sampler.surface-threshold 0.2 \
	--pipeline.model.batching-mode "off" \
	--pipeline.model.batch_size 6144 \
	--pipeline.model.mode semantics \
	--pipeline.model.pretrain False \
	--pipeline.model.feature-generator-config.use-rgb True \
	--pipeline.model.feature-generator-config.use-dir-encoding False \
	--pipeline.model.feature-generator-config.use-pos-encoding True \
	--pipeline.model.feature-generator-config.pos-encoder "sin" \
	--pipeline.model.feature-generator-config.use-normal-encoding False \
	--pipeline.model.feature-generator-config.use-density True \
	--pipeline.model.feature-generator-config.out-rgb-dim 16 \
	--pipeline.model.feature-generator-config.out-density-dim 8 \
	--pipeline.model.feature-generator-config.rot-augmentation True \
	--pipeline.model.space-partitioning "random" \
	--pipeline.model.feature-transformer-model "stratified" \
	--pipeline.model.feature-transformer-stratified-config.grid_size 0.005 \
	--pipeline.model.feature-transformer-stratified-config.quant_size 0.005 \
	--pipeline.model.feature-transformer-stratified-config.load_dir "" \
	nesf-data \
	--data-config $DATA_CONFIG

# needed if we want to use 100% of the pretrained model weights. Just use rgb
# --pipeline.model.feature-transformer-stratified-config.grid_size 0.0054 \
# --pipeline.model.feature-transformer-stratified-config.quant_size 0.001 \
# --pipeline.model.feature-generator-config.out-rgb-dim 3 \
