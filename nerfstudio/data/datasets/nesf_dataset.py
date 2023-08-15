from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.data_utils import (
    get_depth_image_from_path,
    get_semantics_and_mask_tensors_from_path,
)
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.utils.nesf_utils import get_memory_usage


class NesfItemDataset(InputDataset):
    """This is an Input dataset which has the additional metadata information field.
    The meta data field contains the model of this dataset and potentially the semantic masks.
    Args:
        dataparser_outputs: description of where and how to read input data.
        scalefactor: The scaling factor for the dataparser outputs
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1):
        super().__init__(dataparser_outputs, scale_factor)
        assert "model_path" in dataparser_outputs.metadata
        self.model_config = dataparser_outputs.metadata["model_config"]
        self.model_path = dataparser_outputs.metadata["model_path"]
        self.semantics = dataparser_outputs.metadata["semantics"]
        self.mask_indices = torch.tensor(
            [self.semantics.classes.index(mask_class) for mask_class in self.semantics.mask_classes]
        ).view(1, 1, -1)
        
        # get depth image
        self.depth_filenames = self.metadata["depth_filenames"]
        self.depth_unit_scale_factor = self.metadata["depth_unit_scale_factor"]
        self.normal_filenames = self.metadata["normal_filenames"] if "normal_filenames" in self.metadata.keys() else None
        
        
        print("NesfItemDataset - memory usage: ", get_memory_usage())

    def get_metadata(self, data: Dict) -> Dict:
        filepath = self.semantics.filenames[data["image_idx"]]
        semantic_label, mask = get_semantics_and_mask_tensors_from_path(
            filepath=filepath, mask_indices=self.mask_indices, scale_factor=self.scale_factor
        )
        # todo if used add to return value
        # if "mask" in data.keys():
        #     mask = mask & data["mask"]
            
        # Scale depth images to meter units and also by scaling applied to cameras
        filepath = self.depth_filenames[data["image_idx"]]
        height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
        width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])
        scale_factor = self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale
        depth_image = get_depth_image_from_path(
            filepath=filepath, height=height, width=width, scale_factor=scale_factor
        ).float()
        
        metadata = {"model_path": str(self.model_path), "semantics": semantic_label, "depth_image": depth_image}
        if self.normal_filenames is not None:
            normal_filepath = self.normal_filenames[data["image_idx"]]
            normal_image = torch.from_numpy(self.get_numpy_image_from_path(normal_filepath).astype("float32") / 255.0)
            normal_image = normal_image / np.linalg.norm(normal_image, axis=-1, keepdims=True)
            metadata.update({"normal_image": normal_image})
            
        return metadata
    

class NesfDataset(Dataset):
    def __init__(self, datasets: List[NesfItemDataset], main_set: int = 0):
        super().__init__()
        self._datasets: List[NesfItemDataset] = datasets
        self.current_set_idx: int = main_set

    @property
    def has_masks(self):
        return self._datasets[self.current_set_idx].has_masks

    @property
    def scale_factor(self):
        return self._datasets[self.current_set_idx].scale_factor

    @property
    def scene_box(self):
        return self._datasets[self.current_set_idx].scene_box

    @property
    def metadata(self):
        return self._datasets[self.current_set_idx].metadata

    @property
    def cameras(self):
        return self._datasets[self.current_set_idx].cameras

    def __len__(self):
        return len(self._datasets[self.current_set_idx])

    def __iter__(self):
        return iter(self._datasets)

    def __getitem__(self, index) -> Dict:
        return self._datasets[self.current_set_idx][index]

    def get_numpy_image(self, image_idx: int):
        return self._datasets[self.current_set_idx].get_numpy_image(image_idx)

    def get_image(self, image_idx: int):
        return self._datasets[self.current_set_idx].get_image(image_idx)

    def get_data(self, image_idx: int):
        return self._datasets[self.current_set_idx].get_data(image_idx)

    def get_metadata(self, data: Dict):
        return self._datasets[self.current_set_idx].get_metadata(data)

    def image_filenames(self):
        return self._datasets[self.current_set_idx].image_filenames

    def set_current_set(self, dataset_idx: int):
        assert dataset_idx < len(self._datasets)
        assert dataset_idx >= 0

        self.current_set_idx = dataset_idx

    def get_set(self, dataset_idx: int) -> InputDataset:
        assert abs(dataset_idx) < len(self._datasets)
        return self._datasets[dataset_idx]

    def set_count(self) -> int:
        return len(self._datasets)

    @property
    def current_set(self) -> NesfItemDataset:
        return self._datasets[self.current_set_idx]
