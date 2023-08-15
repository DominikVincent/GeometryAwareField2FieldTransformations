import os
import sys
from pathlib import Path
from typing import cast

import torch_geometric

from nerfstudio.configs.method_configs import method_configs
from nerfstudio.models.nesf import NeuralSemanticFieldConfig
from nerfstudio.pipelines.nesf_pipeline import NesfPipelineConfig
from scripts.train import main as train_main


def run_nesf(vis: str = "wandb"):
    # data_config_path = Path("/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/nesf_test_config.json")
    # data_config_path = Path("/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/nesf_test_config_5.json")
    # data_config_path = Path("/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_nesf_train_100.json")
    # data_config_path = Path("/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_nesf_train_10.json")
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_nesf_train_1.json"
    # )
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_nesf_train_1_normals.json"
    # )
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_nesf_train_10.json"
    # )
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klevr-normal_train_10_230.json"
    # )
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_nesf_train_100.json"
    # )

    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_normal_nesf_train_10.json"
    # )
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/toybox-5_nesf_train_10_10.json"
    # )
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/toybox-5_nesf_train_10_270.json"
    # )
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/toybox-5_nesf_train_1_270.json"
    # )
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/toybox-5_nesf_train_100_270.json"
    # )
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/toybox-5_nesf_train_200_270.json"
    # # )
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/toybox-5_nesf_2_train_529_270.json"
    # )
    data_config_path = Path(
        "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/toybox-5_nesf_2_train_500_270.json"
    )
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/toybox-5_nesf_2_train_10_270.json"
    # )

    OUTPUT_DIR = Path("/data/vision/polina/projects/wmh/dhollidt/documents/nerf/nesf_models/")
    DATA_PATH = Path("/data/vision/polina/projects/wmh/dhollidt/datasets/klevr/11")

    trainConfig = method_configs["nesf"]
    trainConfig.pipeline.model = cast(NeuralSemanticFieldConfig, trainConfig.pipeline.model)
    trainConfig.pipeline = cast(NesfPipelineConfig, trainConfig.pipeline)
    # trainConfig = method_configs["nesf_density"]
    trainConfig.vis = vis
    trainConfig.data = DATA_PATH
    trainConfig.output_dir = OUTPUT_DIR
    trainConfig.machine.num_gpus = 1
    trainConfig.pipeline.datamanager.dataparser.data_config = data_config_path
    trainConfig.steps_per_eval_batch = 1000
    trainConfig.steps_per_eval_image = 3000
    trainConfig.steps_per_eval_all_images = 100000000
    trainConfig.max_num_iterations = 10000000
    # trainConfig.load_config = Path("/data/vision/polina/projects/wmh/dhollidt/documents/nerf/nesf_models/tmp/nesf/2023-07-21_103257_286997/auto_eval_config.yml")

    trainConfig.pipeline.datamanager.use_sample_mask = False
    trainConfig.pipeline.datamanager.sample_mask_ground_percentage = 1
    trainConfig.pipeline.datamanager.steps_per_model = 1
    trainConfig.pipeline.datamanager.train_num_images_to_sample_from = 8
    trainConfig.pipeline.datamanager.train_num_times_to_repeat_images = 1
    trainConfig.pipeline.datamanager.eval_num_images_to_sample_from = 8
    trainConfig.pipeline.datamanager.eval_num_times_to_repeat_images = 1
    trainConfig.pipeline.use_3d_mode = False

    trainConfig.pipeline.model.pretrain = False
    trainConfig.pipeline.model.only_last_layer = False
    trainConfig.pipeline.model.mode = "semantics"
    trainConfig.pipeline.model.batching_mode = "off"
    trainConfig.pipeline.model.batch_size = 2048
    trainConfig.pipeline.model.proximity_loss = False
    trainConfig.pipeline.model.feature_generator_config.rot_augmentation = True
    trainConfig.pipeline.model.feature_generator_config.jitter = 0.0005
    trainConfig.pipeline.model.feature_generator_config.jitter_clip =  0.0013
    trainConfig.pipeline.model.feature_generator_config.random_scale =  0.7

    trainConfig.pipeline.model.sampler.surface_sampling = True
    trainConfig.pipeline.model.sampler.get_normals = True
    trainConfig.pipeline.model.sampler.samples_per_ray = 24
    trainConfig.pipeline.model.sampler.ground_removal_mode = "ransac"
    trainConfig.pipeline.model.sampler.ground_tolerance = 0.008
    trainConfig.pipeline.model.sampler.surface_threshold = 0.5
    trainConfig.pipeline.model.sampler.ground_points_count = 1000000
    trainConfig.pipeline.model.sampler.max_points = 16834

    trainConfig.pipeline.model.masker_config.mode = "patch_fp"
    trainConfig.pipeline.model.masker_config.mask_ratio = 0.5
    trainConfig.pipeline.model.masker_config.visualize_masking = False
    trainConfig.pipeline.model.masker_config.num_patches = 100
    trainConfig.pipeline.model.rgb_prediction = "integration"
    trainConfig.pipeline.model.density_prediction = "direct"
    trainConfig.gradient_accumulation_steps = 5

    trainConfig.pipeline.model.use_field2field = False
    trainConfig.pipeline.model.field2field_sampler.surface_sampling = True
    trainConfig.pipeline.model.field2field_sampler.samples_per_ray = 24
    trainConfig.pipeline.model.field2field_sampler.ground_tolerance = 0.008
    trainConfig.pipeline.model.field2field_sampler.surface_threshold = 0.5
    trainConfig.pipeline.model.field2field_sampler.ground_points_count = 100000
    trainConfig.pipeline.model.field2field_sampler.ground_removal_mode = "ransac"

    trainConfig.pipeline.model.feature_generator_config.use_rgb = True
    trainConfig.pipeline.model.feature_generator_config.out_rgb_dim = 3
    trainConfig.pipeline.model.feature_generator_config.use_density = False
    trainConfig.pipeline.model.feature_generator_config.use_pos_encoding = False
    trainConfig.pipeline.model.feature_generator_config.use_dir_encoding = False
    trainConfig.pipeline.model.feature_generator_config.use_normal_encoding = False

    trainConfig.pipeline.model.feature_transformer_model = "stratified"
    trainConfig.pipeline.model.feature_transformer_custom_config.num_layers = 6
    trainConfig.pipeline.model.feature_transformer_custom_config.num_heads = 8
    trainConfig.pipeline.model.feature_transformer_custom_config.dim_feed_forward = 128
    trainConfig.pipeline.model.feature_transformer_custom_config.feature_dim = 128

    trainConfig.pipeline.model.feature_transformer_stratified_config.grid_size = 0.005
    trainConfig.pipeline.model.feature_transformer_stratified_config.window_size = 5
    trainConfig.pipeline.model.feature_transformer_stratified_config.quant_size = 0.0001
    trainConfig.pipeline.model.feature_transformer_stratified_config.num_layers = 4

    trainConfig.pipeline.model.feature_decoder_model = "stratified"
    trainConfig.pipeline.model.feature_decoder_custom_config.num_layers = 2
    trainConfig.pipeline.model.feature_decoder_custom_config.num_heads = 4
    trainConfig.pipeline.model.feature_decoder_custom_config.dim_feed_forward = 128
    trainConfig.pipeline.model.feature_decoder_custom_config.feature_dim = 128

    trainConfig.pipeline.model.feature_decoder_stratified_config.grid_size = 0.005
    trainConfig.pipeline.model.feature_decoder_stratified_config.window_size = 5
    trainConfig.pipeline.model.feature_decoder_stratified_config.quant_size = 0.0001
    trainConfig.pipeline.model.feature_decoder_stratified_config.num_layers = 3
    trainConfig.pipeline.model.feature_decoder_stratified_config.depths = [2,2,4]

    trainConfig.set_timestamp()
    trainConfig.pipeline.datamanager.dataparser.data = trainConfig.data
    trainConfig.pipeline.datamanager.dataparser.train_split_percentage = trainConfig.data
    # trainConfig.pipeline.model.feature_generator_config.visualize_point_batch = True
    # trainConfig.pipeline.model.debug_show_image = True
    trainConfig.save_config()

    train_main(trainConfig)

    # trainer = trainConfig.setup(local_rank=0, world_size=1)
    # trainer.setup()
    # trainer.train()


if __name__ == "__main__":
    print("PID: ", os.getpid())
    args = sys.argv
    run_nesf(args[1])
