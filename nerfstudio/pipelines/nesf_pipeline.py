from __future__ import annotations

import typing
from dataclasses import dataclass, field
from inspect import Parameter
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Type

import numpy as np
import plotly.figure_factory as ff
import torch
import torch.distributed as dist
from PIL import Image
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from typing_extensions import Literal

import wandb
from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.datamanagers.nesf_datamanager import (
    NesfDataManager,
    NesfDataManagerConfig,
    load_model,
)
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.nesf import NeuralSemanticFieldConfig, NeuralSemanticFieldModel
from nerfstudio.pipelines.base_pipeline import (
    Pipeline,
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.nesf_utils import compute_mIoU, get_memory_usage

CONSOLE = Console(width=120)


@dataclass
class NesfPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: NesfPipeline)
    """target class to instantiate"""
    datamanager: NesfDataManagerConfig = NesfDataManagerConfig()
    """specifies the datamanager config"""
    model: NeuralSemanticFieldConfig = NeuralSemanticFieldConfig()
    """specifies the model config"""
    images_per_all_evaluation = 10
    """how many images should be evaluated per scene when evaluating all images. -1 means all"""
    save_images = False
    """save images during all image evaluation"""
    images_to_sample_during_eval_image: int = 8
    use_3d_mode: bool = False


class NesfPipeline(Pipeline):
    """The pipeline class for the nesf nerf setup of multiple cameras for one or a few scenes.

        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    def __init__(
        self,
        config: NesfPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__()
        self.config = config
        self.test_mode = test_mode

        CONSOLE.print("Memory before datamanager init:", get_memory_usage())
        self.datamanager: NesfDataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        CONSOLE.print("Memory after datamanager init:", get_memory_usage())
        self.datamanager.to(device)

        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_datasets is not None, "Missing input dataset"

        self._model: NeuralSemanticFieldModel = config.model.setup(
            scene_box=self.datamanager.train_datasets.get_set(0).scene_box,
            num_train_data=-1,
            metadata=self.datamanager.train_dataset.metadata,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                NeuralSemanticFieldModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        time1  = time()
        ray_bundle, batch = self.datamanager.next_train(step)
        if self.config.use_3d_mode:
            ray_bundle.nears = batch["depth_image"]
            ray_bundle.fars = batch["depth_image"]
        time2 = time()
        CONSOLE.print(f"Time to get next train batch: {time2 - time1}")

        transformer_model_outputs = self.model(ray_bundle, batch)

        time3 = time()
        CONSOLE.print(f"Time to run model forward: {time3 - time2}")
        metrics_dict = self.model.get_metrics_dict(transformer_model_outputs, batch)

        time4 = time()

        # No need for camera opt param groups as the nerfs are assumed to be fixed already.

        loss_dict = self.model.get_loss_dict(transformer_model_outputs, batch, metrics_dict)
        time5 = time()

        CONSOLE.print(f"Time to get metrics dict: {time4 - time3}")
        CONSOLE.print(f"Time to get loss dict: {time5 - time4}")
        torch.cuda.empty_cache()

        return transformer_model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        # self.datamanager.models_to_cpu(step)
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        if self.config.use_3d_mode:
            ray_bundle.nears = batch["depth_image"]
            ray_bundle.fars = batch["depth_image"]
        transformer_model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(transformer_model_outputs, batch)
        loss_dict = self.model.get_loss_dict(transformer_model_outputs, batch, metrics_dict)
        self.train()
        return transformer_model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        if self.config.images_to_sample_during_eval_image > 1:
            image_idx, model_idx, camera_ray_bundle, batch = self.datamanager.next_eval_images(
                step, self.config.images_to_sample_during_eval_image
            )
            if self.config.use_3d_mode:
                for crb, b in zip(camera_ray_bundle, batch):
                    crb.nears = b["depth_image"]
                    crb.fars = b["depth_image"]
            image_idx = image_idx[0]
            model_idx = model_idx[0]
            batch = batch[0]
        else:
            image_idx, model_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)

        # load the model
        model_config = self.datamanager.eval_datasets.get_set(model_idx).model_config
        batch["model"] =  load_model(batch["model_path"], model_config).to(self.device, non_blocking=True)
        batch["image_idx"] = image_idx
        batch["model_idx"] = model_idx
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, batch)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)

        # delte the confusion matrix key as it cant be logged but is needed for eval.py
        metrics_dict.pop('confusion_unnormalized', None)

        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        metrics_dict["model_idx"] = model_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None, save_path: Optional[Path] = None, log_to_wandb=False, miou_3d: bool = False):
        """Iterate over all the images in the eval dataset and get the average.

        Returns:
            metrics_dict: dictionary of metrics
            save_path: path to save the images to. if None, do not save images.
            log_to_wandb: whether to log the images to wandb
            miou_3d: whether to compute the 3d miou. This means one uses the true 3d pointcloud instead of the predicted one. To just judge the quality of the predicted pointcloud.
        """
        wandb_suffix = "_3d" if miou_3d else ""
        if miou_3d:
            self.model.scene_sampler.clear_ground_cache() # type: ignore # pylint: disable=general-type
            surface_sampling = self.model.scene_sampler.config.surface_sampling
            self.model.scene_sampler.config.surface_sampling = False
            samples_per_ray = self.model.scene_sampler.config.samples_per_ray
            self.model.scene_sampler.config.samples_per_ray = 1
        self.eval()
        model_cache = {}
        metrics_dict_list = []
        num_images = (
            min(
                sum(
                    [
                        len(fixed_indices_eval_dataloader)
                        for fixed_indices_eval_dataloader in self.datamanager.fixed_indices_eval_dataloaders
                    ]
                ),
                len(self.datamanager.fixed_indices_eval_dataloaders) * self.config.images_per_all_evaluation,
            )
            if self.config.images_per_all_evaluation >= 0
            else 999999999
        )

        step = wandb.run.step if log_to_wandb is not None else 0
        confusion_matrix = None
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for model_idx, fixed_indices_eval_dataloader in enumerate(self.datamanager.fixed_indices_eval_dataloaders):
                ray_bundles = []
                for image_idx, (camera_ray_bundle, batch) in enumerate(fixed_indices_eval_dataloader):
                    if image_idx >= self.config.images_to_sample_during_eval_image - 1:
                        break
                    if miou_3d:
                        camera_ray_bundle.nears = batch["depth_image"]
                        camera_ray_bundle.fars = batch["depth_image"]
                    ray_bundles.insert(0, camera_ray_bundle)
                for i, (camera_ray_bundle, batch) in enumerate(fixed_indices_eval_dataloader):
                    if miou_3d:
                        camera_ray_bundle.nears = batch["depth_image"]
                        camera_ray_bundle.fars = batch["depth_image"]
                    ray_bundles.insert(0, camera_ray_bundle)
                    batch["model_idx"] = model_idx
                    batch["image_idx"] = batch["image_idx"]
                    print("model_idx", model_idx, "image_idx", batch["image_idx"])

                    if batch["model_path"] not in model_cache:
                        model_config = self.datamanager.eval_datasets.get_set(model_idx).model_config
                        batch["model"] = load_model(batch["model_path"], model_config).to(self.device, non_blocking=True)
                        model_cache[batch["model_path"]] = batch["model"]
                    else:
                        batch["model"] = model_cache[batch["model_path"]].to(self.device, non_blocking=True)
                    batch["model"].to(self.device, non_blocking=True)
                    if i >= self.config.images_per_all_evaluation and self.config.images_per_all_evaluation >= 0:
                        break
                    # time this the following line
                    inner_start = time()
                    height, width = camera_ray_bundle.shape
                    num_rays = height * width
                    outputs = self.model.get_outputs_for_camera_ray_bundle(ray_bundles, batch)
                    metrics_dict, image_dict = self.model.get_image_metrics_and_images(outputs, batch)

                    # move model back to cpu to not waste vram
                    batch["model"].to("cpu", non_blocking=True)

                    assert "num_rays_per_sec" not in metrics_dict
                    metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                    fps_str = "fps"
                    assert fps_str not in metrics_dict
                    metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                    metrics_dict["image_idx"] = batch["image_idx"]
                    metrics_dict["model_idx"] = batch["model_idx"]
                    if "confusion_unnormalized" in metrics_dict:
                        if confusion_matrix is None:
                            confusion_matrix = metrics_dict["confusion_unnormalized"]
                        else:
                            confusion_matrix += metrics_dict["confusion_unnormalized"]

                        metrics_dict.pop("confusion_unnormalized", None)

                    metrics_dict_list.append(metrics_dict)
                    if log_to_wandb:
                        writer.put_dict("test_image"+wandb_suffix, metrics_dict, step=step)

                    if log_to_wandb:
                        for k, img in image_dict.items():
                            writer.put_image("test_image_"+k+wandb_suffix, img, step=step)

                    if "img" in image_dict:
                        img = image_dict["img"]

                        img = img.cpu().numpy()
                        if save_path is not None:
                            file_path = save_path / f"{model_idx:03d}{wandb_suffix}" / f"{batch['image_idx']:04d}.png"
                            # create the directory if it does not exist
                            if not file_path.parent.exists():
                                file_path.parent.mkdir(parents=True)
                            # save the image
                            img_pil = Image.fromarray((img * 255).astype(np.uint8))
                            img_pil.save(file_path)
                    # remove the oldest ray bundle we add so that we have more views in sampling process.
                    ray_bundles.pop()
                    step += 1
                    writer.write_out_storage()
                    progress.advance(task)

        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if key == "image_idx" or key == "model_idx" or key == "confusion_unnormalized":
                continue
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list if not np.isnan(metrics_dict[key])]))
                )

        if confusion_matrix is not None:
            mIoU, mIoU_per_class = compute_mIoU(confusion_matrix)
            metrics_dict["mIoU_total"] = mIoU
            for i, mIoU_class in enumerate(mIoU_per_class):
                metrics_dict[f"mIoU_total_{self.model.semantics.classes[i]}"] = mIoU_class

            if log_to_wandb:
                # normalize confusion matrix
                confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
                fig = ff.create_annotated_heatmap(confusion_matrix, x=self.model.semantics.classes, y=self.model.semantics.classes)
                fig.update_layout(title="Confusion matrix", xaxis_title="Predicted", yaxis_title="Actual")
                wandb.log(
                    {"confusion_matrix_plotly_total"+wandb_suffix: fig},
                    step=wandb.run.step,
                )
        self.train()

        if miou_3d:
            self.model.scene_sampler.clear_ground_cache() # type: ignore # pylint: disable=general-type
            self.model.scene_sampler.config.surface_sampling = surface_sampling
            self.model.scene_sampler.config.samples_per_ray = samples_per_ray

        return metrics_dict

    def load_pipeline(self, loaded_state: Dict[str, Any]) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
        """
        # model = typing.cast(NeuralSemanticFieldModel, self.model)
        # if model.config.feature_transformer_model == "stratified":
        #     load_dir = model.config.feature_transformer_stratified_config.load_dir
        #     if load_dir != "":
        #         missing_keys, unexpected_keys = self.model.feature_transformer.load_state_dict(torch.load(load_dir), strict = False)
        #         print("Loaded feature transformer from pretrained checkpoint")
        #         print("Feature Transformer missing keys", missing_keys)
        #         print("Feature Transformer unexpected keys", unexpected_keys)

        # if model.config.feature_decoder_model == "stratified":
        #     load_dir = model.config.feature_decoder_stratified_config.load_dir
        #     if load_dir != "":
        #         missing_keys, unexpected_keys = self.model.feature_decoder.load_state_dict(torch.load(load_dir), strict = False)
        #         print("Loaded feature decoder from pretrained checkpoint")
        #         print("Feature Decoder missing keys", missing_keys)
        #         print("Feature Decoder unexpected keys", unexpected_keys)

        # TODO questionable if this going to work
        state = {key.replace("module.", ""): value for key, value in loaded_state.items()}
        missing_keys, unexpected_keys = self.load_state_dict(state, strict=False)

        print("Loaded Nesf pipeline from checkpoint")
        print("Missing keys", missing_keys)
        print("Unexpected keys", unexpected_keys)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> typing.List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks

    def get_param_groups(self) -> Dict[str, typing.List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        # TODO(ethan): assert that key names don't overlap
        return {**datamanager_params, **model_params}
