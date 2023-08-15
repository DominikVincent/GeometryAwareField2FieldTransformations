#!/bin/bash

source /data/vision/polina/projects/wmh/dhollidt/conda/bin/activate
conda activate nerfstudio2

# DATA_CONFIG="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_nesf_train_1.json"
# DATA_CONFIG="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_nesf_train_10.json"
# DATA_CONFIG="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_nesf_train_100.json"
DATA_CONFIG="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/toybox-5_nesf_2_train_500_270.json"
# DATA_CONFIG="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/toybox-5_nesf_train_10_270.json"

# RAYS=131072
RAYS=65536
# RAYS=40960
# RAYS=32768
# RAYS=16384

# export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

ns-train nesf --data /data/vision/polina/projects/wmh/dhollidt/datasets/klevr_nesf/0  \
	--output-dir /data/vision/polina/projects/wmh/dhollidt/documents/nerf/nesf_models/ \
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
	--pipeline.datamanager.eval-num-rays-per-batch $RAYS \
	--pipeline.datamanager.use-sample-mask False \
    --pipeline.datamanager.sample-mask-ground-percentage 1.0 \
	--pipeline.model.eval-num-rays-per-chunk $RAYS \
	--pipeline.model.sampler.surface-sampling True \
	--pipeline.model.sampler.samples-per-ray 24 \
	--pipeline.model.sampler.ground_removal_mode "ransac" \
	--pipeline.model.sampler.ground-points-count 5000000 \
	--pipeline.model.sampler.ground-tolerance 0.008 \
	--pipeline.model.sampler.surface-threshold 0.5 \
	--pipeline.model.sampler.get-normals False \
	--pipeline.model.batching-mode "off" \
	--pipeline.model.batch_size 2048 \
	--pipeline.model.mode rgb \
	--pipeline.model.pretrain True  \
	--pipeline.model.feature-generator-config.use-rgb True \
	--pipeline.model.feature-generator-config.use-dir-encoding True \
	--pipeline.model.feature-generator-config.use-pos-encoding True \
	--pipeline.model.feature-generator-config.use-density True \
	--pipeline.model.feature-generator-config.out-rgb-dim 16 \
	--pipeline.model.feature-generator-config.pos-encoder "sin" \
	--pipeline.model.feature-generator-config.out-density-dim 8 \
	--pipeline.model.feature-generator-config.rot-augmentation True \
	--pipeline.model.space-partitioning "random" \
	--pipeline.model.feature-transformer-model "stratified" \
	--pipeline.model.feature-transformer-stratified-config.grid_size 0.005 \
	--pipeline.model.feature-transformer-stratified-config.quant_size 0.0001 \
	--pipeline.model.feature-transformer-stratified-config.window_size 4 \
	--pipeline.model.feature-transformer-stratified-config.load_dir "" \
	--pipeline.model.feature_decoder_model "stratified" \
	--pipeline.model.feature-decoder-stratified-config.grid_size 0.005 \
	--pipeline.model.feature-decoder-stratified-config.quant_size 0.0001 \
	--pipeline.model.feature-decoder-stratified-config.window_size 4 \
	--pipeline.model.feature-decoder-stratified-config.load_dir "" \
	--pipeline.model.feature-decoder-stratified-config.num-layers 3 \
	--pipeline.model.feature-decoder-stratified-config.depths 2 2 4 \
	--pipeline.model.masker_config.mask_ratio 0.5 \
	--pipeline.model.masker_config.mode "patch" \
	--pipeline.model.masker_config.num-patches 100 \
	--pipeline.model.rgb-prediction "integration" \
	nesf-data \
	--data-config $DATA_CONFIG
