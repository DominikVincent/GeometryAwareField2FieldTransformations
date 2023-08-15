#!/bin/bash

source /data/vision/polina/projects/wmh/dhollidt/conda/bin/activate
conda activate nerfstudio2

ns-train nesf_density --data /data/vision/polina/projects/wmh/dhollidt/datasets/klevr/11 \
	--output-dir /data/vision/polina/projects/wmh/dhollidt/documents/nerf/nesf_models/ \
	--vis wandb \
	--pipeline.datamanager.steps-per-model 1 \
	--pipeline.model.mode density \
	--pipeline.model.density_prediction direct \
	--pipeline.model.pretrain True  \
	--pipeline.model.mask_ratio 0.6  \
	--pipeline.model.use-feature-rgb True \
	--pipeline.model.use-feature-dir True \
	--pipeline.model.use-feature-pos True \
	--pipeline.model.use-feature-density True \
	--pipeline.model.rgb_feature_dim 16 \
	--pipeline.model.space_partitioning "evenly" \
	--pipeline.model.feature_transformer_num_layers 8 \
	--pipeline.model.feature_transformer_num_heads 16 \
	--pipeline.model.feature_transformer_dim_feed_forward 64 \
	--pipeline.model.feature_transformer_dropout_rate 0.1 \
	--pipeline.model.feature_transformer_feature_dim 64 \
	--pipeline.model.decoder_feature_transformer_num_layers 2 \
	--pipeline.model.decoder_feature_transformer_num_heads 2 \
	--pipeline.model.decoder_feature_transformer_dim_feed_forward 32 \
	--pipeline.model.decoder_feature_transformer_dropout_rate 0.1 \
	--pipeline.model.decoder_feature_transformer_feature_dim 32 \
	nesf-data \
	--data-config /data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/nesf_test_config_5.json
