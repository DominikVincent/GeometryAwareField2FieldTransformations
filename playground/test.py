
# import all the necessary modules
from pathlib import Path
import random
import os

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

def _load_checkpoint(load_dir, load_step, data_dir: Path, local_rank: int = 0, world_size: int=1) -> None:
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(
                data=data_dir
                ),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    )

    device = "cpu" if world_size == 0 else f"cuda:{local_rank}"
    pipeline = pipeline.setup(
            device=device, test_mode="inference", world_size=world_size, local_rank=local_rank
    )

    """Helper function to load pipeline and optimizer from prespecified checkpoint"""
    if load_dir is not None:
        if load_step is None:
            print("Loading latest checkpoint from load_dir")
            # NOTE: this is specific to the checkpoint name format
            load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
        load_path = load_dir / f"step-{load_step:09d}.ckpt"
        assert load_path.exists(), f"Checkpoint {load_path} does not exist"
        loaded_state = torch.load(load_path, map_location="cpu")
        # load the checkpoints for pipeline, optimizers, and gradient scalar
        pipeline.load_pipeline(loaded_state["pipeline"])
        print(f"done loading checkpoint from {load_path}")
    else:
        print("No checkpoints to load, training from scratch")
    return pipeline


# Dice DATA
MODEL_CHECKPOINT_PATH = Path("/data/vision/polina/projects/wmh/dhollidt/documents/nerf/outputs/dice_256/nerfacto/2023-01-16_101826/nerfstudio_models")
MODEL_LOAD_STEP = 24000
DATA_PATH = Path("/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/dice_rand_v3")

pipeline = _load_checkpoint(MODEL_CHECKPOINT_PATH, MODEL_LOAD_STEP, DATA_PATH)
model = pipeline.model