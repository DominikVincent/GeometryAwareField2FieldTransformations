# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Put all the method implementations in one location.
"""

from __future__ import annotations

from typing import Dict

import tyro
from nerfacc import ContractionType

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import LoggingConfig, ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.datamanagers.depth_datamanager import DepthDataManagerConfig
from nerfstudio.data.datamanagers.nesf_datamanager import NesfDataManagerConfig
from nerfstudio.data.datamanagers.semantic_datamanager import SemanticDataManagerConfig
from nerfstudio.data.datamanagers.variable_res_datamanager import (
    VariableResDataManagerConfig,
)
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.friends_dataparser import FriendsDataParserConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import (
    InstantNGPDataParserConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.nesf_dataparser import NesfDataParserConfig
from nerfstudio.data.dataparsers.phototourism_dataparser import (
    PhototourismDataParserConfig,
)
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import SchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.model_components.nesf_components import (
    FeatureGeneratorTorchConfig,
    PointNetWrapperConfig,
    SceneSamplerConfig,
    TranformerEncoderModelConfig,
)
from nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.nesf import NeuralSemanticFieldConfig
from nerfstudio.models.semantic_nerfw import SemanticNerfWModelConfig
from nerfstudio.models.tensorf import TensoRFModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.pipelines.nesf_pipeline import NesfPipelineConfig

method_configs: Dict[str, TrainerConfig] = {}
descriptions = {
    "nerfacto": "Recommended real-time model tuned for real captures. This model will be continually updated.",
    "depth-nerfacto": "Nerfacto with depth supervision.",
    "instant-ngp": "Implementation of Instant-NGP. Recommended real-time model for unbounded scenes.",
    "instant-ngp-bounded": "Implementation of Instant-NGP. Recommended for bounded real and synthetic scenes",
    "mipnerf": "High quality model for bounded scenes. (slow)",
    "semantic-nerfw": "Predicts semantic segmentations and filters out transient objects.",
    "vanilla-nerf": "Original NeRF model. (slow)",
    "tensorf": "tensorf",
    "dnerf": "Dynamic-NeRF model. (slow)",
    "phototourism": "Uses the Phototourism data.",
    "nesf": "Neural Semantic Field model. prototype",
}


method_configs["nerfacto"] = TrainerConfig(
    method_name="nerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["depth-nerfacto"] = TrainerConfig(
    method_name="depth-nerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=DepthDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=DepthNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["instant-ngp"] = TrainerConfig(
    method_name="instant-ngp",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=DynamicBatchPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=NerfstudioDataParserConfig(), train_num_rays_per_batch=8192),
        model=InstantNGPModelConfig(eval_num_rays_per_chunk=8192),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        }
    },
    viewer=ViewerConfig(num_rays_per_chunk=64000),
    vis="viewer",
)


method_configs["instant-ngp-bounded"] = TrainerConfig(
    method_name="instant-ngp-bounded",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=DynamicBatchPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=InstantNGPDataParserConfig(), train_num_rays_per_batch=8192),
        model=InstantNGPModelConfig(
            eval_num_rays_per_chunk=8192,
            contraction_type=ContractionType.AABB,
            render_step_size=0.001,
            max_num_samples_per_ray=48,
            near_plane=0.01,
            background_color="black",
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        }
    },
    viewer=ViewerConfig(num_rays_per_chunk=64000),
    vis="viewer",
)


method_configs["mipnerf"] = TrainerConfig(
    method_name="mipnerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=NerfstudioDataParserConfig(), train_num_rays_per_batch=1024),
        model=VanillaModelConfig(
            _target=MipNerfModel,
            loss_coefficients={"rgb_loss_coarse": 0.1, "rgb_loss_fine": 1.0},
            num_coarse_samples=128,
            num_importance_samples=128,
            eval_num_rays_per_chunk=1024,
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        }
    },
)

method_configs["semantic-nerfw"] = TrainerConfig(
    method_name="semantic-nerfw",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=SemanticDataManagerConfig(
            dataparser=FriendsDataParserConfig(), train_num_rays_per_batch=4096, eval_num_rays_per_batch=8192
        ),
        model=SemanticNerfWModelConfig(eval_num_rays_per_chunk=1 << 16),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 16),
    vis="viewer",
)

method_configs["vanilla-nerf"] = TrainerConfig(
    method_name="vanilla-nerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BlenderDataParserConfig(),
        ),
        model=VanillaModelConfig(_target=NeRFModel),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
        "temporal_distortion": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
    },
)

method_configs["tensorf"] = TrainerConfig(
    method_name="tensorf",
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BlenderDataParserConfig(),
        ),
        model=TensoRFModelConfig(),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=0.001),
            "scheduler": SchedulerConfig(lr_final=0.0001, max_steps=30000),
        },
        "encodings": {
            "optimizer": AdamOptimizerConfig(lr=0.02),
            "scheduler": SchedulerConfig(lr_final=0.002, max_steps=30000),
        },
    },
)

method_configs["dnerf"] = TrainerConfig(
    method_name="dnerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=DNeRFDataParserConfig()),
        model=VanillaModelConfig(
            _target=NeRFModel,
            enable_temporal_distortion=True,
            temporal_distortion_params={"kind": TemporalDistortionKind.DNERF},
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
        "temporal_distortion": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
    },
)


method_configs["phototourism"] = TrainerConfig(
    method_name="phototourism",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VariableResDataManagerConfig(  # NOTE: one of the only differences with nerfacto
            dataparser=PhototourismDataParserConfig(),  # NOTE: one of the only differences with nerfacto
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)


# RAYS_PER_BATCH = 131072
RAYS_PER_BATCH = 65536
# RAYS_PER_BATCH = 40000
# RAYS_PER_BATCH = 32768
# RAYS_PER_BATCH = 24576
# RAYS_PER_BATCH=16384
# RAYS_PER_BATCH=8192
# RAYS_PER_BATCH=4096
# RAYS_PER_BATCH=131072
MAX_NUM_ITERATIONS = 500000
END_DECAY = 80000
LR_START = 0.005
LR_END = LR_START * 0.1
TRANSFORMER_LR_SCALE = 0.1
method_configs["nesf"] = TrainerConfig(
    method_name="nesf",
    experiment_name="/tmp",
    steps_per_eval_batch=100,
    steps_per_eval_image=250,
    steps_per_save=5000,
    max_num_iterations=MAX_NUM_ITERATIONS,
    steps_per_eval_all_images=1000000,
    mixed_precision=False,
    pipeline=NesfPipelineConfig(
        datamanager=NesfDataManagerConfig(
            dataparser=NesfDataParserConfig(),
            train_num_rays_per_batch=RAYS_PER_BATCH,
            eval_num_rays_per_batch=RAYS_PER_BATCH,
            steps_per_model=1,
            train_num_images_to_sample_from=8,
            train_num_times_to_repeat_images=1,
            eval_num_images_to_sample_from=8,
            eval_num_times_to_repeat_images=4,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeuralSemanticFieldConfig(eval_num_rays_per_chunk=RAYS_PER_BATCH, mode="semantics", pretrain=False),
    ),
    optimizers={
        "feature_network": {
            "optimizer": AdamOptimizerConfig(lr=LR_START, eps=1e-13),
            "scheduler": SchedulerConfig(lr_final=LR_END, max_steps=END_DECAY),
            # "scheduler": None,
        },
        "feature_transformer": {
            "optimizer": AdamOptimizerConfig(lr=LR_START, eps=1e-13),
            "scheduler": SchedulerConfig(lr_final=LR_END, max_steps=END_DECAY),
            # "scheduler": None,
        },
        "feature_transformer_transformer": {
            "optimizer": AdamOptimizerConfig(lr=LR_START*TRANSFORMER_LR_SCALE, eps=1e-13),
            "scheduler": SchedulerConfig(lr_final=LR_END*TRANSFORMER_LR_SCALE, max_steps=END_DECAY),
            # "scheduler": None,
        },
        "learned_low_density_params": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-13),
            "scheduler": SchedulerConfig(lr_final=1e-4, max_steps=END_DECAY),
            # "scheduler": None,
        },
        "decoder": {
            "optimizer": AdamOptimizerConfig(lr=LR_START, eps=1e-13),
            "scheduler": SchedulerConfig(lr_final=LR_END, max_steps=END_DECAY),
            # "scheduler": None,
        },
        "decoder_transformer": {
            "optimizer": AdamOptimizerConfig(lr=LR_START*TRANSFORMER_LR_SCALE, eps=1e-13),
            "scheduler": SchedulerConfig(lr_final=LR_END*TRANSFORMER_LR_SCALE, max_steps=END_DECAY),
            # "scheduler": None,
        },
        "head": {
            "optimizer": AdamOptimizerConfig(lr=LR_START, eps=1e-13),
            "scheduler": SchedulerConfig(lr_final=LR_END, max_steps=END_DECAY),
            # "scheduler": None,
        },
        "field_transformer": {
            "optimizer": AdamOptimizerConfig(lr=4E-5, eps=1e-13),
            "scheduler": SchedulerConfig(lr_final=1E-6, max_steps=END_DECAY),
            # "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=64, websocket_port=7017, quit_on_train_completion=False),
    save_only_latest_checkpoint=True,
    vis="viewer",
    logging=LoggingConfig(steps_per_log=10),
)


method_configs["nesf_density"] = TrainerConfig(
    method_name="nesf",
    experiment_name="/tmp",
    steps_per_eval_batch=100,
    steps_per_eval_image=250,
    steps_per_save=5000,
    max_num_iterations=50000,
    steps_per_eval_all_images=1000000,
    mixed_precision=False,
    pipeline=NesfPipelineConfig(
        datamanager=NesfDataManagerConfig(
            dataparser=NesfDataParserConfig(),
            train_num_rays_per_batch=64,
            eval_num_rays_per_batch=64,
            steps_per_model=1,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeuralSemanticFieldConfig(
            eval_num_rays_per_chunk=256,
            mode="density",
            feature_generator_config=FeatureGeneratorTorchConfig(
                use_rgb=True,
                out_rgb_dim=16,
                use_density=False,
                out_density_dim=8,
                use_dir_encoding=True,
                use_pos_encoding=True,
                visualize_point_batch=False
            ),
            sampler=SceneSamplerConfig(),
            pretrain=True,
            feature_transformer_custom_config=TranformerEncoderModelConfig(
                num_layers=4,
                num_heads=8,
                dim_feed_forward=64,
                dropout_rate=0.1,
                feature_dim=64,
            ),
            feature_decoder_custom_config=TranformerEncoderModelConfig(
                num_layers=2,
                num_heads=2,
                dim_feed_forward=32,
                dropout_rate=0.1,
                feature_dim=32,
            ),
            density_prediction="direct",
        ),
    ),
    optimizers={
        "feature_network": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-13),
            "scheduler": SchedulerConfig(lr_final=1e-5, max_steps=30000),
            # "scheduler": None,
        },
        "feature_transformer": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-13),
            "scheduler": SchedulerConfig(lr_final=1e-5, max_steps=30000),
            # "scheduler": None,
        },
        "learned_low_density_params": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-13),
            "scheduler": SchedulerConfig(lr_final=1e-5, max_steps=30000),
            # "scheduler": None,
        },
        "decoder": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-13),
            "scheduler": SchedulerConfig(lr_final=1e-5, max_steps=30000),
            # "scheduler": None,
        },
        "head": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-13),
            "scheduler": SchedulerConfig(lr_final=1e-5, max_steps=30000),
            # "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=64, websocket_port=7011, quit_on_train_completion=False),
    save_only_latest_checkpoint=False,
    vis="viewer",
    logging=LoggingConfig(steps_per_log=10),
)

# method_configs["nesf"] = TrainerConfig(
#     method_name="nesf",
#     experiment_name="/tmp",
#     steps_per_eval_batch=50,
#     steps_per_eval_image=500,
#     steps_per_save=500,
#     max_num_iterations=10000,
#     mixed_precision=False,
#     pipeline=NesfPipelineConfig(
#         datamanager=NesfDataManagerConfig(
#             dataparser=NesfDataParserConfig(),
#             train_num_rays_per_batch=32,
#             eval_num_rays_per_batch=32,
#             camera_optimizer=CameraOptimizerConfig(
#                 mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
#             ),
#         ),
#         model=NeuralSemanticFieldConfig(eval_num_rays_per_chunk=64, rgb=False),
#     ),
#     optimizers={
#         "feature_network": {
#             "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
#             "scheduler": None,
#         },
#         "feature_transformer": {
#             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             "scheduler": None,
#         },
#         "learned_low_density_params": {
#             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             "scheduler": None,
#         },
#     },
#     viewer=ViewerConfig(num_rays_per_chunk=64, websocket_port=7011),
#     save_only_latest_checkpoint=False,
#     vis="viewer",
# )


AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=method_configs, descriptions=descriptions)
    ]
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
