from pathlib import Path
import random

from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.configs.experiment_config import ExperimentConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig
import torch

from utils import *


# Dice DATA
MODEL_CHECKPOINT_PATH = Path("/YOUR/PATH/HERE/outputs/dice_256/nerfacto/2023-01-16_101826/nerfstudio_models")
MODEL_LOAD_STEP = 24000
DATA_PATH = Path("/YOUR/PATH/HERE/data/dice_rand_v3")

# CLEVR DATA
# MODEL_CHECKPOINT_PATH = Path("/data/vision/polina/scratch/clintonw/datasets/nerfacto/0/nerfacto/2023-01-13_145424/nerfstudio_models")
# MODEL_LOAD_STEP = 29999
# DATA_PATH = Path("/data/vision/polina/scratch/clintonw/datasets/kubric/0/")

OUTPUT_DIR = Path("/YOUR/PATH/HERE/playground")


trainConfig = TrainerConfig(
    method_name="nerfacto",
    experiment_name="/tmp",
    data=DATA_PATH,
    output_dir=OUTPUT_DIR,
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
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
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
    vis="wandb",
    load_dir=MODEL_CHECKPOINT_PATH,
    load_step=MODEL_LOAD_STEP
)

trainConfig.set_timestamp()
trainConfig.pipeline.datamanager.dataparser.data = trainConfig.data
trainConfig.save_config()

trainer = trainConfig.setup(local_rank=0, world_size=1)
trainer.setup()

trainer.train()