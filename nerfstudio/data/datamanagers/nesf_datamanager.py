from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import plotly.express as px
import torch
from rich.console import Console
from torch.nn import Parameter
from torch.utils.data import DataLoader, Dataset
from typing_extensions import Literal

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    AnnotatedDataParserUnion,
    DataManager,
)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.nesf_dataparser import NerfstudioDataParserConfig, Nesf
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.datasets.nesf_dataset import NesfDataset, NesfItemDataset
from nerfstudio.data.pixel_samplers import EquirectangularPixelSampler, PixelSampler
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.misc import get_dict_to_torch
from nerfstudio.utils.nesf_utils import get_memory_usage

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600


@dataclass
class NesfDataManagerConfig(InstantiateConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: NesfDataManager)
    """Target class to instantiate."""
    dataparser: AnnotatedDataParserUnion = NerfstudioDataParserConfig()
    """Specifies the dataparser used to unpack the data."""
    train_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per training iteration."""
    train_num_images_to_sample_from: int = 4
    """Number of images to sample during training iteration."""
    train_num_times_to_repeat_images: int = 4
    """When not training on all images, number of iterations before picking new
    images. If -1, never pick new images."""
    eval_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per eval iteration."""
    eval_num_images_to_sample_from: int = 4
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = 4
    """When not evaluating on all images, number of iterations before picking
    new images. If -1, never pick new images."""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the image indices to use during eval; if None, uses all."""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()
    """Specifies the camera pose optimizer used during training. Helpful if poses are noisy, such as for data from
    Record3D."""
    collate_fn = staticmethod(nerfstudio_collate)
    """Specifies the collate function to use for the train and eval dataloaders."""
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    steps_per_model: int = 1
    """Number of steps one model is queried before the next model is queried. The models are taken sequentially."""
    use_sample_mask: bool = False
    """If yes it will generate a sampling mask based on the semantic map and only sample non ground pixels plus some randomly sampled ground pixels"""
    sample_mask_ground_percentage: float = 1.0
    """The number of ground pixels to sample in the sampling mask randomly set to true. 1.0 means all pixels uniformly."""


class NesfDataManager(DataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: NesfDataManagerConfig
    train_datasets: NesfDataset
    eval_datasets: NesfDataset
    train_dataparser_outputs: DataparserOutputs
    train_pixel_sampler: Optional[PixelSampler] = None
    eval_pixel_sampler: Optional[PixelSampler] = None

    def __init__(
        self,
        config: NesfDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.last_model = None
        self.last_model_idx = None
        self.last_eval_model = None
        self.last_eval_model_idx = None
        CONSOLE.print(f"Datamanager mem usage: ", get_memory_usage())
        self.dataparser: Nesf = self.config.dataparser.setup()
        CONSOLE.print(f"Datamanager after setup usage: ", get_memory_usage())
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")
        CONSOLE.print(f"Datamanager mem usage: ", get_memory_usage())
        self.train_datasets = self.create_train_datasets()
        CONSOLE.print(f"Datamanager mem usage create_train_datasets: ", get_memory_usage())
        self.eval_datasets = self.create_eval_datasets()
        CONSOLE.print(f"Datamanager mem usage create_eval_datasets: ", get_memory_usage())
        self.train_dataset = self.train_datasets
        self.eval_dataset = self.eval_datasets
        self.eval_image_model = 0
        super().__init__()

    def create_train_datasets(self) -> NesfDataset:
        """Sets up the data loaders for training"""
        return NesfDataset(
            [
                NesfItemDataset(dataparser_outputs=dataparser_output, scale_factor=self.config.camera_res_scale_factor)
                for dataparser_output in self.train_dataparser_outputs
            ]
        )

    def create_eval_datasets(self) -> NesfDataset:
        """Sets up the data loaders for evaluation"""
        return NesfDataset(
            [
                NesfItemDataset(dataparser_outputs=dataparser_output, scale_factor=self.config.camera_res_scale_factor)
                for dataparser_output in self.dataparser.get_dataparser_outputs(split=self.test_split)
            ]
        )

    def _get_pixel_sampler(  # pylint: disable=no-self-use
        self, dataset: NesfItemDataset, *args: Any, **kwargs: Any
    ) -> PixelSampler:
        """Infer pixel sampler to use."""
        # If all images are equirectangular, use equirectangular pixel sampler
        is_equirectangular = dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value
        if is_equirectangular.all():
            return EquirectangularPixelSampler(*args, **kwargs)
        # Otherwise, use the default pixel sampler
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")
        return PixelSampler(*args, **kwargs)

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_datasets is not None
        CONSOLE.print("Setting up training dataset...")

        self.meta_train_image_dataloader = PrefetchLoader(
            self.train_datasets,
            batch_size=self.config.train_num_images_to_sample_from,
            prefetch_batches=20,
            collate_fn=self.config.collate_fn,
            device=self.device,
        )

        self.train_pixel_samplers = [
            self._get_pixel_sampler(train_dataset, self.config.train_num_rays_per_batch)
            for train_dataset in self.train_datasets
        ]

        def get_camera_conf(group_name) -> CameraOptimizerConfig:
            self.config.camera_optimizer.param_group = group_name
            return deepcopy(self.config.camera_optimizer)

        self.train_camera_optimizers = [
            get_camera_conf(group_name=get_dir_of_path(dataparser_output.image_filenames[0])).setup(
                num_cameras=train_dataset.cameras.shape[0], device=self.device
            )
            for dataparser_output, train_dataset in zip(self.train_dataparser_outputs, self.train_datasets)
        ]

        self.train_ray_generators = [
            RayGenerator(
                train_dataset.cameras.to(self.device),
                train_camera_optimizer,
            )
            for train_dataset, train_camera_optimizer in zip(self.train_datasets, self.train_camera_optimizers)
        ]
        CONSOLE.print(f"Datamanager mem usage end of train setup: ", get_memory_usage())

        return

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_datasets is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloaders = [
            CacheDataloader(
                eval_dataset,
                num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
                num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
                device=self.device,
                num_workers=self.world_size * 4,
                pin_memory=True,
                collate_fn=self.config.collate_fn,
            )
            for eval_dataset in self.eval_datasets
        ]

        self.iter_eval_image_dataloaders = [
            iter(eval_image_dataloader) for eval_image_dataloader in self.eval_image_dataloaders
        ]

        self.meta_eval_image_dataloader = PrefetchLoader(
            self.eval_datasets,
            batch_size=self.config.train_num_images_to_sample_from,
            prefetch_batches=2,
            collate_fn=self.config.collate_fn,
            device=self.device,
        )

        print("iters created")
        self.eval_pixel_samplers = [
            self._get_pixel_sampler(eval_dataset, self.config.eval_num_rays_per_batch)
            for eval_dataset in self.eval_datasets
        ]
        self.eval_ray_generators = [
            RayGenerator(
                eval_dataset.cameras.to(self.device),
                train_camera_optimizer,  # should be shared between train and eval.
            )
            for eval_dataset, train_camera_optimizer in zip(self.eval_datasets, self.train_camera_optimizers)
        ]

        # for loading full images, used for all image evaluation
        self.fixed_indices_eval_dataloaders = [
            FixedIndicesEvalDataloader(
                input_dataset=eval_dataset,
                device=self.device,
                num_workers=self.world_size * 4,
            )
            for eval_dataset in self.eval_datasets
        ]

        self.eval_dataloaders = [
            RandIndicesEvalDataloader(
                input_dataset=eval_dataset,
                image_indices=self.config.eval_image_indices,
                device=self.device,
                num_workers=self.world_size * 4,
            )
            for eval_dataset in self.eval_datasets
        ]

        CONSOLE.print(f"Datamanager mem usage end of eval setup: ", get_memory_usage())

    def debug_stats(self):
        non_gpu = []
        for i, dataset in enumerate(self.train_datasets):
            dataset = cast(NesfItemDataset, dataset)
            # if dataset.model.device.type != "cpu":
            #     non_gpu.append(i)
            print(i, dataset.model.device)
        print("Non gpu: ", non_gpu)

    def models_to_cpu(self, step):
        """Moves all models who shouldnt be active to cpu."""
        model_idx = self.step_to_dataset(step)
        for i, dataset in enumerate(self.train_datasets):
            # print(i, model_idx, dataset.model.device)
            if i == model_idx:
                continue

            dataset = cast(NesfItemDataset, dataset)
            if dataset.model.device.type != "cpu":
                dataset.model.to("cpu", non_blocking=True)

    def move_model_to_cpu(self, dataset):
        dataset.model.to("cpu", non_blocking=True)

    def models_to_cpu_threading(self, step):
        """Moves all models who shouldn't be active to CPU."""
        model_idx = self.step_to_dataset(step)
        for i, dataset in enumerate(self.train_datasets):
            if i != model_idx:
                dataset = cast(NesfItemDataset, dataset)
                thread = threading.Thread(target=self.move_model_to_cpu, args=(dataset,))
                thread.start()

    @profiler.time_function
    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        if self.last_model is not None:
            with self.meta_train_image_dataloader.lock:
                if self.last_model_idx not in self.meta_train_image_dataloader.queued_idx:
                    self.last_model.to("cpu", non_blocking=True)

        self.train_count += 1
        time1 = time.time()
        # image_batch: dict = next(self.meta_train_image_dataloader)
        image_batch: dict = next(self.meta_train_image_dataloader)
        CONSOLE.print("Currently queued (", len(self.meta_train_image_dataloader.queued_idx),"/", self.meta_train_image_dataloader.prefetch_batches,"): ", self.meta_train_image_dataloader.queued_idx)
        CONSOLE.print("jobs queued up in threadpool:", self.meta_train_image_dataloader.executor._work_queue.qsize())
        assert image_batch["model"][0].device != "cpu"
        model_idx = image_batch["model_idx"]
        del image_batch["model_idx"]
        self.last_model = image_batch["model"][0]
        self.last_model_idx = model_idx
        print("Processing :", model_idx)

        # model_idx = self.step_to_dataset(step)
        # image_batch: dict = next(self.iter_train_image_dataloaders[model_idx])


        time2 = time.time()
        if self.config.use_sample_mask:
            semantic_mask = image_batch["semantics"]  # [N, H, W]
            # count the number of pixels per class
            # pixels_per_class = torch.bincount(semantic_mask.flatten(), minlength=13)
            # CONSOLE.print("Class counts percentage: ", (pixels_per_class/semantic_mask.numel()).p)

            mask = semantic_mask >= 1

            # add random noise to the mask
            total_pixels = mask.numel()
            num_true = int(total_pixels * self.config.sample_mask_ground_percentage)

            # Create a flattened tensor of shape (total_pixels,)
            flat_mask = torch.zeros(total_pixels, dtype=torch.bool)

            # Randomly select indices to set as True
            indices = torch.randperm(total_pixels)[:num_true]
            flat_mask[indices] = True

            # Reshape the flattened tensor back to the desired shape
            mask_salt_and_pepper = flat_mask.reshape(mask.shape)

            # mask_background = mask_salt_and_pepper & (semantic_mask == 0)
            # print("mask_background ratio: ", (mask_background.sum() / mask_background[0].numel()).item())

            mask = mask | mask_salt_and_pepper

            image_batch["mask"] = mask

            # mask_d = mask.detach().cpu().numpy()[0].squeeze()
            # fig = px.imshow(mask_d)
            # fig.show()
            # fig = px.imshow(image_batch["image"][0])
            # fig.show()

        assert self.train_pixel_samplers[model_idx] is not None
        batch = self.train_pixel_samplers[model_idx].sample(image_batch)
        time3 = time.time()
        ray_indices = batch["indices"]
        batch["model_idx"] = model_idx
        batch["model"] = image_batch["model"][0]
        ray_bundle = self.train_ray_generators[model_idx](ray_indices)
        time4 = time.time()
        assert str(batch["image"].device) == "cpu"
        assert str(batch["semantics"].device) == "cpu"
        assert str(batch["indices"].device) == "cpu"

        CONSOLE.print(f"Next Train - get_batch: {time2 - time1}")
        CONSOLE.print(f"Next Train - sample pixels: {time3 - time2}")
        CONSOLE.print(f"Next Train - ray generation: {time4 - time3}")
        CONSOLE.print("After next train mem usage: ", get_memory_usage())

        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        if self.last_eval_model is not None:
            with self.meta_eval_image_dataloader.lock:
                if self.last_eval_model_idx not in self.meta_eval_image_dataloader.queued_idx:
                    self.last_eval_model.to("cpu", non_blocking=True)

        image_batch: dict = next(self.meta_eval_image_dataloader)
        assert image_batch["model"][0].device != "cpu"
        model_idx = image_batch["model_idx"]
        del image_batch["model_idx"]
        self.last_eval_model = image_batch["model"][0]
        self.last_eval_model_idx = model_idx
        CONSOLE.print(f"Eval model scene {model_idx}")

        assert self.eval_pixel_samplers[model_idx] is not None
        batch = self.eval_pixel_samplers[model_idx].sample(image_batch)
        ray_indices = batch["indices"]
        batch["model_idx"] = model_idx
        batch["model"] = image_batch["model"][0]
        ray_bundle = self.eval_ray_generators[model_idx](ray_indices)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, int, RayBundle, Dict]:
        model_idx = self.eval_image_model % self.eval_datasets.set_count()
        self.eval_image_model += 1

        for camera_ray_bundle, batch in self.eval_dataloaders[model_idx]:
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            return image_idx, model_idx, camera_ray_bundle, batch
        raise ValueError("No more eval images")

    def next_eval_images(self, step: int, images: int) -> Tuple[List[int], List[int], List[RayBundle], List[Dict]]:
        model_idx = self.eval_image_model % self.eval_datasets.set_count()
        self.eval_image_model += 1

        image_idxs = []
        model_idxs = []
        ray_bundles = []
        batches = []
        for _ in range(images):
            for camera_ray_bundle, batch in self.eval_dataloaders[model_idx]:
                assert camera_ray_bundle.camera_indices is not None
                image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])

                image_idxs.append(image_idx)
                model_idxs.append(model_idx)
                ray_bundles.append(camera_ray_bundle)
                batches.append(batch)

        return image_idxs, model_idxs, ray_bundles, batches

    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        # TODO consider whether this is needed as the models parameters are assumed to be fixed. Potentially return {}
        param_groups = {}
        for train_camera_optimizer in self.train_camera_optimizers:
            camera_opt_params = list(train_camera_optimizer.parameters())
            if train_camera_optimizer.config.mode != "off":
                assert len(camera_opt_params) > 0
                param_groups[self.config.camera_optimizer.param_group] = camera_opt_params
            else:
                assert len(camera_opt_params) == 0

        return param_groups

    def step_to_dataset(self, step: int) -> int:
        """Returns the dataset index for the given step."""
        return (step // self.config.steps_per_model) % self.train_datasets.set_count()

    def steps_to_next_dataset(self, step: int) -> int:
        """Returns the number of steps until the next dataset is used."""
        return self.config.steps_per_model - (step % self.config.steps_per_model)


def get_dir_of_path(path: Path) -> str:
    return str(path.parent.name)



class PrefetchLoader:
    def __init__(self, datasets, batch_size, prefetch_batches, collate_fn, device):
        self.model_configs = [datasets.get_set(i).model_config for i in range(datasets.set_count()) ]
        self.datasets = cycle(enumerate(datasets))
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.executor = ThreadPoolExecutor(max_workers=prefetch_batches)
        self.queue = Queue()
        self.collate_fn = collate_fn
        self.device = device
        self.lock = Lock()
        self.queued_idx = []  # to check the existence of a dataset

    def __iter__(self):
        return self

    def prefetch(self):
        idx, dataset = next(self.datasets)
        loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)
        batch = next(iter(loader))
        batch["model_idx"] = idx
        model_config = self.model_configs[batch["model_idx"]]
        batch["model"] = [load_model(batch["model_path"][0], model_config)] * self.batch_size
        del batch["model_path"]
        batch = get_dict_to_torch(batch, device=self.device, exclude=["image", "semantics"])
        with self.lock:
            self.queue.put(batch)
            self.queued_idx.append(idx)

    def __next__(self):
        time1 = time.time()
        if self.queue.qsize() < self.prefetch_batches and self.executor._work_queue.qsize() < self.prefetch_batches:
            batches_to_prefetch = max(self.prefetch_batches - self.queue.qsize(), 0)
            print("Prefetching", batches_to_prefetch, "batches queue size", self.queue.qsize())
            futures = [self.executor.submit(self.prefetch) for _ in range(batches_to_prefetch)]
            # futures = [self.prefetch() for _ in range(self.prefetch_batches - self.queue.qsize())]

        time2 = time.time()
        batch = self.queue.get()
        with self.lock:
            self.queued_idx.remove(batch["model_idx"])

        return batch

def load_model(model_path, model_config):
    time1 = time.time()
    pred_normals = "normal" in str(model_path)

    model=DepthNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15,
                                    predict_normals=pred_normals,
                                    **model_config)


    # scene box will be loaded from state
    # num_train_data is 271 for our models, not relevant during eval anyway
    scene_box = SceneBox(aabb = torch.zeros((2,3)))

    num_train_data = 231 if "klevr-normal" in str(model_path) else 271
    model = model.setup(scene_box=scene_box,
            num_train_data=num_train_data,
            metadata={})
    time2 = time.time()
    loaded_state = torch.load(model_path, map_location="cpu")["pipeline"]

    state = {key.replace("module.", ""): value for key, value in loaded_state.items()}
    state = {key.replace("_model.", ""): value for key, value in state.items()}
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=True)
    assert missing_keys == []
    assert unexpected_keys == []

    time3 = time.time()
    print("Load Model - loading model took", time2 - time1)
    print("Load Model - loading state dict took", time3 - time2)


    return model