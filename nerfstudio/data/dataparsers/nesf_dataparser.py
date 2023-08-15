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
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Type, cast

import torch
from rich.console import Console

from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
    Semantics,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.models.base_model import Model
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.nesf_utils import CLASS_TO_COLOR, get_memory_usage
from nerfstudio.utils.writer import put_config

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600

SEMANTIC_CLASSES_CLEVR_OBJECTS = ["background", "Cube", "Cylinder", "Sphere", "Torus", "Gear"]
SEMANTIC_CLASSES_KUBASIC_OBJECTS = [
    "background",
    "cube",
    "cylinder",
    "sphere",
    "cone",
    "torus",
    "gear",
    "torus_knot",
    "sponge",
    "spot",
    "teapot",
    "suzanne",
]
SEMANTIC_CLASSES_TOYBOX_5 = ["background", "airplane", "car", "chair", "sofa", "table"]
SEMANTIC_CLASSES_TOYBOX_13 = ["background", "airplane", "bench", "cabinet", "car", "chair", "display", "lamp", "loudspeaker", "rifle", "sofa", "table", "telephone", "vessel"]



# TODO delete if not needed
# @dataclass
# class NesfDataparserOutputs:
#     """Dataparser outputs for the which will be used by the DataManager
#     for creating RayBundle and RayGT objects."""
#
#     image_filenames: List[List[Path]]
#     """Filenames for the images."""
#     cameras: List[Cameras]
#     """Camera object storing collection of camera information in dataset."""
#     alpha_color: Optional[List[TensorType[3]]] = None
#     """Color of dataset background."""
#     scene_box: List[SceneBox] = field(default_factory=lambda: [SceneBox()])
#     """Scene box of dataset. Used to bound the scene or provide the scene scale depending on model."""
#     mask_filenames: Optional[List[List[Path]]] = None
#     """Filenames for any masks that are required"""
#     metadata: Dict[str, Any] = to_immutable_dict({})
#     """Dictionary of any metadata that be required for the given experiment.
#     Will be processed by the InputDataset to create any additional tensors that may be required.
#     """
#     dataparser_transform: List[TensorType[3, 4]] = torch.eye(4)[:3, :]
#     """Transform applied by the dataparser."""
#     dataparser_scale: List[float] = 1.0
#     """Scale applied by the dataparser."""
#
#     def as_dict(self) -> dict:
#         """Returns the dataclass as a dictionary."""
#         return vars(self)
#
#     def save_dataparser_transform(self, path: Path):
#         """Save dataparser transform to json file. Some dataparsers will apply a transform to the poses,
#         this method allows the transform to be saved so that it can be used in other applications.
#
#         Args:
#             path: path to save transform to
#         """
#         data = {
#             "transform": [transform.tolist() for transform in self.dataparser_transform],
#             "scale": [float(scale) for scale in self.dataparser_scale],
#         }
#         if not path.parent.exists():
#             path.parent.mkdir(parents=True)
#         with open(path, "w", encoding="UTF-8") as file:
#             json.dump(data, file, indent=4)


@dataclass
class NesfDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: Nesf)

    data_config: Path = Path("")
    """Path to the config of the Nesf data. It's a json {config:[{model_config: config, data_parser_config: config, load_step: 1, load_dir:1}, 
    ...]}"""


@dataclass
class Nesf(DataParser):
    """Nerfstudio DatasetParser"""

    config: NesfDataParserConfig
    model_cache: Dict[str, Model] = field(default_factory=lambda: {})

    def _generate_dataparser_outputs(self, split="train") -> List[DataparserOutputs]:
        # pylint: disable=too-many-statements
        CONSOLE.print("Memory usage start dataparsing: ", get_memory_usage())
        if self.config.data_config.suffix == ".json":
            data_config = load_from_json(self.config.data_config)
        else:
            data_config = load_from_json(self.config.data_config / "data_config.json")

        put_config("config", data_config, 0)
        
        CONSOLE.print("Memory usage: ", get_memory_usage())
        
        data_parser_outputs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = []
            for conf in data_config["config"]:
                future = executor.submit(self.process_conf, conf, split)
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                data_parser_outputs.append(future.result())
        # models = []
        # data_parser_outputs = []
        # for conf in data_config["config"]:
        #     print("split: ", split)
        #     print("conf:", conf["data_parser_config"])
        #     print(conf.get("set_type", "train"))

        #     nerfstudio = NerfstudioDataParserConfig(**conf["data_parser_config"]).setup()
        #     # TODO find a more global solution for casting instead of just one key
        #     nerfstudio.config.data = Path(nerfstudio.config.data)

        #     # dataparser_output = nerfstudio.get_dataparser_outputs(split=conf.get("set_type", "train"))
        #     if "set_type" in conf:
        #         split = conf["set_type"]
        #     dataparser_output = nerfstudio.get_dataparser_outputs(split=split)
        #     CONSOLE.print(f"[green] loaded dataparser_output from {nerfstudio.config.data}")
        #     print(len(dataparser_output.image_filenames))
        #     print([int(path.name[5:-4]) for path in dataparser_output.image_filenames])

        #     # models.append({"load_dir": conf["load_dir"], "load_step": conf["load_step"], "data_parser": nerfstudio})
        #     # TODO maybe load model

        #     # parent path of file
        #     data_path = dataparser_output.image_filenames[0].parent.resolve()
        #     CONSOLE.print(f"Loading model from {data_path}")
        #     model = self._load_model(
        #         load_dir=Path(conf["load_dir"]),
        #         load_step=conf["load_step"],
        #         data_dir=data_path,
        #         config=conf["model_config"],
        #     ).to("cpu").to(torch.float32)
        #     CONSOLE.print(f"[green] loaded model from {data_path}")

        #     # get the list of semantic images. For each image there should be a semantic image.
        #     semantic_paths = []
        #     for image_filename in dataparser_output.image_filenames:
        #         image_name = image_filename.stem
        #         base_path = image_filename.parent
        #         number = image_name.split("_")[1]
        #         semantic_paths.append(base_path / f"segmentation_{number}.png")

        #     semantics = Semantics(
        #         filenames=semantic_paths, classes=SEMANTIC_CLASSES_CLEVR_OBJECTS, colors=CLASS_TO_COLOR, mask_classes=[]
        #     )
        #     # TODO update dataparser_output.metadata with model
        #     dataparser_output.metadata.update({"model": model, "semantics": semantics})

        #     data_parser_outputs.append(dataparser_output)

        # TODO remove if uneeded
        # dataparser_outputs = NesfDataparserOutputs([data_parser_output.image_filenames for data_parser_output in data_parser_outputs],
        #     cameras=[data_parser_output.cameras for data_parser_output in data_parser_outputs],
        #     scene_box=[data_parser_output.scene_box for data_parser_output in data_parser_outputs],
        #     mask_filenames=[data_parser_output.mask_filenames if len(data_parser_output.mask_filenames) > 0 else None for data_parser_output in data_parser_outputs],
        #     dataparser_scale=[data_parser_output.dataparser_scale for data_parser_output in data_parser_outputs],
        #     dataparser_transform=[data_parser_output.dataparser_scale for data_parser_output in data_parser_outputs],
        #     metadata={
        #         # TODO safe model here
        #     },
        # )
        CONSOLE.print("Returned NESF dataparser outputs")
        data_parser_outputs.sort(key=lambda x: int(x.image_filenames[0].parent.name))
        return data_parser_outputs
    
    
    def process_conf(self, conf, split):
        print("split: ", split)
        print("conf:", conf["data_parser_config"])
        print(conf.get("set_type", "train"))
        CONSOLE.print("Current mem usage before parsing: ", get_memory_usage())
        

        nerfstudio = NerfstudioDataParserConfig(**conf["data_parser_config"]).setup()
        # TODO find a more global solution for casting instead of just one key
        nerfstudio.config.data = Path(nerfstudio.config.data)

        if "set_type" in conf:
            split = conf["set_type"]
        dataparser_output = nerfstudio.get_dataparser_outputs(split=split)
        
        CONSOLE.print(f"[green] loaded dataparser_output from {nerfstudio.config.data}")
        print(len(dataparser_output.image_filenames))
        print([int(path.name[5:-4]) for path in dataparser_output.image_filenames])

        # models.append({"load_dir": conf["load_dir"], "load_step": conf["load_step"], "data_parser": nerfstudio})
        # TODO maybe load model

        # parent path of file
        data_path = dataparser_output.image_filenames[0].parent.resolve()
        CONSOLE.print(f"Loading model from {data_path}")
        model_path = self._load_model(
            load_dir=Path(conf["load_dir"]),
            load_step=conf["load_step"])

        # get the list of semantic images. For each image, there should be a semantic image.
        semantic_paths = []
        for image_filename in dataparser_output.image_filenames:
            image_name = image_filename.stem
            base_path = image_filename.parent
            number = image_name.split("_")[1]
            semantic_paths.append(base_path / f"segmentation_{number}.png")

        
        if "klevr" in conf["data_parser_config"]["data"]:
            classes = SEMANTIC_CLASSES_CLEVR_OBJECTS
        elif "toybox-5" in conf["data_parser_config"]["data"]:
            classes = SEMANTIC_CLASSES_TOYBOX_5
        elif "toybox-13" in conf["data_parser_config"]["data"]:
            classes = SEMANTIC_CLASSES_TOYBOX_13
        else:
            raise ValueError("Dataset not supported")
        semantics = Semantics(
            filenames=semantic_paths, classes=classes, colors=CLASS_TO_COLOR, mask_classes=[]
        )
        # TODO update dataparser_output.metadata with model
        dataparser_output.metadata.update({"model_path": model_path, "model_config": conf["model_config"], "semantics": semantics})

        CONSOLE.print("Current mem usage after parsing: ", get_memory_usage())
        return dataparser_output

    def _load_model(
        self, load_dir, load_step
    ) -> Path:

        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        if load_dir is not None:
            if load_step is None:
                print("Loading latest checkpoint from load_dir")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
        else:
            raise ValueError("No checkpoints to load, training from scratch")

        return load_path
