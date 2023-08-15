import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil, sqrt
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import lovely_tensors as lt
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.graph_objs as go
import torch
import torch.nn.functional as F
import tyro
from rich.console import Console
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.classification import ConfusionMatrix
from typing_extensions import Literal

import wandb
from nerfstudio.cameras.rays import RayBundle, stack_ray_bundles
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import pred_normal_loss
from nerfstudio.model_components.nesf_components import *
from nerfstudio.model_components.renderers import (
    NormalsRenderer,
    RGBRenderer,
    SemanticRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.nesf_utils import *
from nerfstudio.utils.writer import put_config

lt.monkey_patch()

CONSOLE = Console(width=120)


@dataclass
class NeuralSemanticFieldConfig(ModelConfig):
    """Config for Neural Semantic field"""

    _target: Type = field(default_factory=lambda: NeuralSemanticFieldModel)

    background_color: Literal["random", "last_sample"] = "last_sample"
    """Whether to randomize the background color."""

    mode: Literal["rgb", "semantics", "density", "normals", "normals,rgb"] = "rgb"
    """The mode in which the model is trained. It predicts whatever mode is chosen. Density is only used for pretraining. In ztheory any combinations of the modes is also possible."""

    sampler: SceneSamplerConfig = SceneSamplerConfig()
    """The sampler used in the model."""

    feature_generator_config: FeatureGeneratorTorchConfig = FeatureGeneratorTorchConfig()
    """The feature generating model to use."""

    # feature_transformer_config: AnnotatedTransformerUnion = PointNetWrapperConfig()
    # dirty workaround because Union of configs didnt work
    feature_transformer_model: Literal["pointnet", "custom", "stratified"] = "custom"
    feature_transformer_pointnet_config: PointNetWrapperConfig = PointNetWrapperConfig()
    feature_transformer_custom_config: TranformerEncoderModelConfig = TranformerEncoderModelConfig()
    feature_transformer_stratified_config: StratifiedTransformerWrapperConfig = StratifiedTransformerWrapperConfig()

    # In case of pretraining we use a decoder together with a linear unit as prediction head.
    feature_decoder_model: Literal["pointnet", "custom", "stratified"] = "custom"
    feature_decoder_pointnet_config: PointNetWrapperConfig = PointNetWrapperConfig()
    feature_decoder_custom_config: TranformerEncoderModelConfig = TranformerEncoderModelConfig()
    feature_decoder_stratified_config: StratifiedTransformerWrapperConfig = StratifiedTransformerWrapperConfig()
    """If pretraining is used, what should the encoder look like"""

    use_field2field: bool = False
    field2field_sampler: SceneSamplerConfig = SceneSamplerConfig()
    """Sampling the neural field does not require the same sampling strategy as the feature transformer. Hence we can use a different sampler for the field transformer."""
    field2field_config: FieldTransformerConfig = FieldTransformerConfig()
    """Whether to use a field transformer or not. If yes, what should it look like. It allows to get a true field to field mapping."""

    masker_config: MaskerConfig = MaskerConfig()
    """If pretraining is used the masker will be used mask poi"""

    pretrain: bool = False
    """Flag indicating whether the model is in pretraining mode or not."""
    
    only_last_layer: bool = False
    """Whether to only train the last layer of the model or not."""

    space_partitioning: Literal["row_wise", "random", "evenly"] = "random"
    """How to partition the image space when rendering."""

    density_prediction: Literal["direct", "integration"] = "direct"
    """How to train the density prediction. With the direct nerf density output or throught the integration process"""
    density_cutoff: float = 2e4
    """Large density values might be an issue for training. Hence they can get cut off with this."""
    
    rgb_prediction: Literal["direct", "integration"] = "integration"
    """How to train the rgb prediction. With the direct nerf density output or throught the integration process"""

    batching_mode: Literal["sequential", "random", "sliced", "off"] = "off"
    """Usually all samples are fed into the transformer at the same time. This could be too much for the model to understand and also too much for VRAM.
    Hence we batch the samples:
     - sequential: we batch the samples by wrapping them sequentially into batches.
     - random: take random permuatations of points for batching.
     - sliced: Sort points by x coordinate and then slice them into batches.
     - off: no batching is done."""
    batch_size: int = 1536
    
    proximity_loss: bool = False
    """Whether to use the proximity loss or not. Encourages that close points give similar predicitons."""
    proximity_loss_mult: float = 0.01
    """The multiplier for the proximity loss."""

    debug_show_image: bool = False
    """Show the generated image."""
    debug_show_final_point_cloud: bool = True
    """Show the generated image."""
    log_confusion_to_wandb: bool = True
    plot_confusion: bool = False
    visualize_semantic_pc: bool = False


def get_wandb_histogram(tensor):
    def create_histogram(tensor):
        hist, bin_edges = np.histogram(tensor.detach().cpu().numpy(), bins=100)
        return hist, bin_edges

    tensor = tensor.flatten()
    hist = create_histogram(tensor)
    return wandb.Histogram(np_histogram=hist)


class NeuralSemanticFieldModel(Model):
    config: NeuralSemanticFieldConfig

    def __init__(self, config: NeuralSemanticFieldConfig, metadata: Dict, **kwargs) -> None:
        assert "semantics" in metadata.keys() and isinstance(metadata["semantics"], Semantics)
        self.semantics: Semantics = metadata["semantics"]
        self.broken_normals = {}
        if not any(mode in config.mode for mode in ["rgb", "semantics", "density", "normals"]):
            raise ValueError(f"Unknown mode {self.config.mode}")
        super().__init__(config=config, **kwargs)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        return []

    def populate_modules(self):
        print(self.config.sampler)
        # Losses
        self.rgb_loss = torch.nn.L1Loss()
        # self.rgb_loss = torch.nn.MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(
                # weight=torch.tensor([1.0, 32.0, 32.0, 32.0, 32.0, 32.0]),
                reduction="mean"
            )
        self.density_loss = torch.nn.L1Loss()

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.confusion = ConfusionMatrix(task="multiclass", num_classes=len(self.semantics.classes), normalize="true")
        self.confusion_non_normalized = ConfusionMatrix(task="multiclass", num_classes=len(self.semantics.classes), normalize="none")

        self.scene_sampler: SceneSampler = self.config.sampler.setup()
        self.field_transformer_sampler: SceneSampler = self.config.field2field_sampler.setup()

        # Feature extractor
        self.feature_model: FeatureGeneratorTorch = self.config.feature_generator_config.setup(aabb=self.scene_box.aabb)

        # Feature Transformer
        semantic_classes_count = len(self.semantics.classes)

        # restrict to positive values if predicting density or rgb
        if self.config.feature_transformer_model == "pointnet":
            self.feature_transformer = self.config.feature_transformer_pointnet_config.setup(
                input_size=self.feature_model.get_out_dim(),
                activation=torch.nn.ReLU(),
            )
        elif self.config.feature_transformer_model == "custom":
            self.feature_transformer = self.config.feature_transformer_custom_config.setup(
                input_size=self.feature_model.get_out_dim(),
                activation=torch.nn.ReLU(),
            )
        elif self.config.feature_transformer_model == "stratified":
            self.feature_transformer = self.config.feature_transformer_stratified_config.setup(
                input_size=self.feature_model.get_out_dim(),
                activation=torch.nn.ReLU(),
            )
        else:
            raise ValueError(f"Unknown feature transformer config {self.config.feature_transformer_model}")

        if self.config.pretrain:
            self.masker: Masker = self.config.masker_config.setup(output_size=self.feature_transformer.get_out_dim())

        if self.config.pretrain:
            # TODO add transformer decoder
            if self.config.feature_decoder_model == "pointnet":
                self.decoder = self.config.feature_decoder_pointnet_config.setup(
                    input_size=self.feature_transformer.get_out_dim(),
                    activation=torch.nn.ReLU(),
                )
            elif self.config.feature_decoder_model == "custom":
                self.decoder = self.config.feature_decoder_custom_config.setup(
                    input_size=self.feature_transformer.get_out_dim(),
                    activation=torch.nn.ReLU(),
                )
            elif self.config.feature_decoder_model == "stratified":
                self.decoder = self.config.feature_decoder_stratified_config.setup(
                    input_size=self.feature_transformer.get_out_dim(),
                    activation=torch.nn.ReLU(),
                )
            else:
                raise ValueError(f"Unknown feature transformer config {self.config.feature_decoder_model}")

            if "rgb" in self.config.mode:
                self.head_rgb = torch.nn.Sequential(
                    torch.nn.Linear(self.decoder.get_out_dim(), 3),
                    torch.nn.ReLU(),
                )
            if "semantics" in self.config.mode:
                self.head_semantics = torch.nn.Sequential(
                    torch.nn.Linear(self.decoder.get_out_dim(), semantic_classes_count),
                    torch.nn.Softmax(dim=1),
                )
            if "density" in self.config.mode:
                self.head_density = torch.nn.Sequential(
                    torch.nn.Linear(self.decoder.get_out_dim(), 1),
                    torch.nn.ReLU(),
                )
            if "normals" in self.config.mode:
                self.head_normals = torch.nn.Sequential(
                    torch.nn.Linear(self.decoder.get_out_dim(), 3),
                    torch.nn.Tanh(),
                )
          
        else:
            self.decoder = torch.nn.Identity()

            self.head_semantics = torch.nn.Sequential(
                torch.nn.Linear(self.feature_transformer.get_out_dim(), 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, semantic_classes_count)
            )
        # self.learned_low_density_value = torch.nn.Parameter(torch.randn(output_size) * 0.1 + 0.8)
        self.learned_low_density_value_semantics = torch.nn.Parameter(torch.zeros(semantic_classes_count) * 0.1 + 0.8, requires_grad=False)
        self.learned_low_density_value_semantics[0] = 1.0

        self.learned_low_density_value_rgb = torch.nn.Parameter(torch.zeros(3) * 0.1 + 0.8, requires_grad=False)
        self.learned_low_density_value_normals = torch.nn.Parameter(torch.randn(3) * 0.1 + 0.8, requires_grad=True)
        self.learned_low_density_value_density = torch.nn.Parameter(torch.randn(1) * 0.1 + 0.8, requires_grad=True)


        if self.config.use_field2field:
            self.field_transformer = self.config.field2field_config.setup(input_size=self.feature_transformer.get_out_dim())

        # Renderer
        head_params = {}
        if "rgb" in self.config.mode:
            self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
            head_params["rgb"] = sum(p.numel() for p in self.head_rgb.parameters())
        if "semantics" in self.config.mode:
            self.renderer_semantics = SemanticRenderer()
            head_params["semantics"] = sum(p.numel() for p in self.head_semantics.parameters())
        if "density" in self.config.mode:
            self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
            head_params["density"] = sum(p.numel() for p in self.head_density.parameters())
        if "normals" in self.config.mode:
            self.renderer_normals = NormalsRenderer()
            head_params["normals"] = sum(p.numel() for p in self.head_normals.parameters())
        if not any(mode in self.config.mode for mode in ["rgb", "semantics", "density", "normals"]):
            raise ValueError(f"Unknown mode {self.config.mode}")

        # This model gets used if no model gets passed in the batch, e.g. when using the viewer
        self.fallback_model: Optional[Model] = None

        # count parameters
        total_params = sum(p.numel() for p in self.parameters())
        
        put_config(
            "network parameters",
            {
                "feature_generator_parameters": sum(p.numel() for p in self.feature_model.parameters()),
                "feature_transformer_parameters": sum(p.numel() for p in self.feature_transformer.parameters()),
                "decoder_parameters": sum(p.numel() for p in self.decoder.parameters()),
                **head_params,
                "field_transformer_parameters": sum(p.numel() for p in self.field_transformer.parameters()) if self.config.use_field2field else 0,

                "total_parameters": total_params,
            },
            0,
        )
        CONSOLE.print("Feature Generator has", sum(p.numel() for p in self.feature_model.parameters()), "parameters")
        CONSOLE.print(
            "Feature Transformer has", sum(p.numel() for p in self.feature_transformer.parameters()), "parameters"
        )
        CONSOLE.print("Decoder has", sum(p.numel() for p in self.decoder.parameters()), "parameters")
        CONSOLE.print("Field Transformer has", sum(p.numel() for p in self.field_transformer.parameters()), "parameters")
        CONSOLE.print("The number of NeSF parameters is: ", total_params)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        masker_param_list = list(self.masker.parameters()) if self.config.pretrain else []
        
        feature_transformer_params = []
        feature_transformer_transformer_params = [] 
        if self.config.feature_transformer_model == "stratified":
            if self.config.only_last_layer:
                parameters_to_optimize = []
                excluded_layers = ['layers.0', 'layers.1', 'layers.2']
                for name, param in self.feature_transformer.named_parameters():
                    include_parameter = True
                    # Check if the parameter's name contains any excluded layers
                    for excluded_layer in excluded_layers:
                        if excluded_layer in name:
                            include_parameter = False
                            param.requires_grad = False
                            break

                    # Add the parameter to the list if it is not in an excluded layer
                    if include_parameter:
                        parameters_to_optimize.append((name, param))
            
            
            else:
                parameters_to_optimize = list(self.feature_transformer.named_parameters())
            
            feature_transformer_params = [p for n, p in parameters_to_optimize if "blocks" not in n and p.requires_grad]
            feature_transformer_transformer_params = [p for n, p in parameters_to_optimize if "blocks" in n and p.requires_grad]
            
        else:
            feature_transformer_transformer_params = list(self.feature_transformer.parameters())
            
        decoder_params = []
        decoder_transformer_params = []
        if self.config.feature_decoder_model == "stratified":
            decoder_params = [p for n, p in self.decoder.named_parameters() if "blocks" not in n and p.requires_grad]
            decoder_transformer_params = [p for n, p in self.decoder.named_parameters() if "blocks" in n and p.requires_grad]
        else:
            decoder_transformer_params = list(self.decoder.parameters())
        
        head_parameters = []
        if "rgb" in self.config.mode:
            head_parameters += list(self.head_rgb.parameters())
        if "semantics" in self.config.mode:
            head_parameters += list(self.head_semantics.parameters())
        if "density" in self.config.mode:
            head_parameters += list(self.head_density.parameters())
        if "normals" in self.config.mode:
            head_parameters += list(self.head_normals.parameters())

        param_groups = {
            "feature_network": list(self.feature_model.parameters()),
            "feature_transformer": feature_transformer_params,
            "feature_transformer_transformer": feature_transformer_transformer_params,
            "learned_low_density_params": [self.learned_low_density_value_density, self.learned_low_density_value_normals, self.learned_low_density_value_rgb, self.learned_low_density_value_semantics] + masker_param_list,
            "decoder": decoder_params,
            "decoder_transformer": decoder_transformer_params,
            "head": head_parameters,
            "field_transformer": list(self.field_transformer.parameters()) if self.config.use_field2field else [],
        }

        # filter the empty ones
        for key in list(param_groups.keys()):
            if len(param_groups[key]) == 0:
                del param_groups[key]

        return param_groups

    def check_broken_normals(self, field_outputs_raw, model_idx):
        normals = field_outputs_raw[FieldHeadNames.NORMALS]
        normals_pred = field_outputs_raw[FieldHeadNames.PRED_NORMALS]
        normals_std = torch.std(normals, dim=0)
        normals_pred_std = torch.std(normals_pred, dim=0)
        normal_total_std = torch.sum(normals_std)
        normal_pred_total_std = torch.sum(normals_pred_std)
        
        CONSOLE.print("Total std of      normals: ", normal_total_std)
        CONSOLE.print("Total std of pred_normals: ", normal_pred_total_std)
        if normal_pred_total_std < 0.1:
            self.broken_normals[model_idx] = True

    def get_neural_point_cloud(self, ray_bundle: RayBundle, model, model_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            ray_samples,
            weights,
            field_outputs_raw,
            density_mask,
            original_fields_outputs,
        ) = self.scene_sampler.sample_scene(ray_bundle, model, model_idx)
        all_samples_count = density_mask.shape[0]*density_mask.shape[1]

        field_outputs_raw, transform_batch = self.batching(ray_samples, field_outputs_raw)

        outs, transform_batch = self.feature_model(field_outputs_raw, transform_batch)  # 1, low_dense, 49t

        # assert outs is not nan or not inf
        assert not torch.isnan(outs).any()
        assert not torch.isinf(outs).any()

        # assert outs is not nan or not inf
        assert not torch.isnan(outs).any()
        assert not torch.isinf(outs).any()

        if self.config.pretrain:
            outs, mask, ids_restore, batch = self.masker.mask(outs, transform_batch)
            outs = self.feature_transformer(outs, batch=transform_batch)
            outs, transform_batch = self.masker.unmask(outs, transform_batch, ids_restore)
            field_encodings = self.decoder(outs, batch=transform_batch)
        else:
            field_encodings = self.feature_transformer(outs, batch=transform_batch)
        
        return field_encodings, transform_batch["points_xyz_orig"] 
    
    def rot_augmentation(self, query_points, neural_points):
        angles_np = np.random.uniform(0, 2 * np.pi, size=(1,)).astype('f')
        angles = torch.from_numpy(angles_np)
        
        # Construct the rotation matrices from the random angles.
        zeros = torch.zeros_like(angles)
        ones = torch.ones_like(angles)
        c = torch.cos(angles)
        s = torch.sin(angles)
        
        
        rot_matrix = torch.stack(
            [
                torch.stack([c, -s, zeros], dim=-1),
                torch.stack([s, c, zeros], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1),
            ],
            dim=-2,
        ).to(query_points.device).squeeze(0)

        query_points = torch.matmul(query_points, rot_matrix)
        neural_points = torch.matmul(neural_points, rot_matrix)
        return query_points, neural_points


    @profiler.time_function
    def get_outputs(self, ray_bundle: RayBundle, batch: Union[Dict[str, Any], None] = None):
        model: Model = self.get_model(batch)
        if "neural_points" in batch and "neural_features" in batch:
            neural_points = batch["neural_points"]
            neural_features = batch["neural_features"]
        else:
            time1=time.time()
            neural_features, neural_points = self.get_neural_point_cloud(batch["ray_bundle_np"], model, batch["model_idx"])
            neural_features = neural_features.squeeze(0)
            neural_points = neural_points.squeeze(0)
            time2 = time.time()
            CONSOLE.print("Neural point cloud generation took", time2-time1, "seconds")
        (
            ray_samples,
            weights,
            field_outputs_raw,
            density_mask,
            original_fields_outputs,
        ) = self.field_transformer_sampler.sample_scene(ray_bundle, model, batch["model_idx"])

        time3 = time.time()
        all_samples_count = density_mask.shape[0]*density_mask.shape[1]
        query_points = ray_samples.frustums.get_positions()
        query_points = query_points.reshape(-1, 3)
        if self.training and self.feature_model.config.rot_augmentation:
            query_points, neural_points = self.rot_augmentation(query_points, neural_points)
        timea=time.time()
        field_encodings = self.field_transformer(query_points, neural_features, neural_points)

        # total_time = 0
        # repetitions = 10
        # if query_points.shape[0] >= 16384:
        #     points_xyz_save = query_points.clone()
        #     query_points = query_points[:16384]
        #     fake_features = torch.randn((16384, 48)).cuda()
        #     for _ in range(repetitions):
        #         with torch.no_grad():
        #             starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        #             starter.record()
        #             field_encodings_2 = self.field_transformer(query_points, fake_features, query_points)
        #             ender.record()
        #             torch.cuda.synchronize()  # synchronize CUDA operations
        #             curr_time = starter.elapsed_time(ender)/1000
        #             total_time += curr_time
        #     query_points = points_xyz_save
        #     CONSOLE.print("CUDA: Forward - feature transformer: ", total_time/repetitions)
        field_encodings = field_encodings.unsqueeze(0)
        timeb = time.time()
        print("Querying ", query_points.shape, "took", timeb-timea, "seconds")
        time4 = time.time()
        CONSOLE.print("Field2Field Transformer took", time4-time3, "seconds")

        CONSOLE.print("Field Transformer Ray samples used", weights.shape[0], "out of", all_samples_count, "samples")
        outputs = {
            "neural_points": neural_points,
            "neural_features": neural_features,
        }
        field_outputs_dict = {}
        if self.config.pretrain:
            # probably does not work (yet)
            if "rgb" in self.config.mode:
                field_outputs_dict["rgb"] = self.head_rgb(field_encodings)
            if "density" in self.config.mode:
                field_outputs_dict["density"] = self.head_density(field_encodings)
            if "semantics" in self.config.mode:
                field_outputs_dict["semantics"] = self.head_semantics(field_encodings)
            if "normals" in self.config.mode:
                field_outputs_dict["normals"] = self.head_normals(field_encodings)
        else:
            field_outputs_dict["semantics"] = self.head_semantics(field_encodings)
            
            if self.config.proximity_loss:
                query_points_noise = query_points + torch.randn_like(query_points) * 0.003
                field_encodings_noise = self.field_transformer(query_points_noise, neural_features, neural_points)
                field_encodings_noise = field_encodings_noise.unsqueeze(0)
                outputs["pointcloud_pred_noise"] = field_encodings_noise
                outputs["pointcloud_pred"] = field_encodings

        if "rgb" in self.config.mode:
            # debug rgb
            rgb = torch.empty((*density_mask.shape, 3), device=self.device)
            weights_all = torch.zeros((*density_mask.shape, 1), device=self.device)  # 64, 48, 6

            rgb[density_mask] = field_outputs_dict["rgb"]
            weights_all[density_mask] = weights

            rgb[~density_mask] = torch.nn.functional.relu(self.learned_low_density_value_rgb)
            weights_all[~density_mask] = 0.01

            rgb_integrated = self.renderer_rgb(rgb, weights=weights_all)
            outputs["rgb"] = rgb_integrated
            rgb_gt = original_fields_outputs[FieldHeadNames.RGB]
            outputs["rgb_gt"] = rgb_gt
            outputs["rgb_pred"] = rgb
        if "semantics" in self.config.mode:
            semantics = torch.zeros((*density_mask.shape, len(self.semantics.classes)), device=self.device)  # 64, 48, 6
            weights_all = torch.zeros((*density_mask.shape, 1), device=self.device)  # 64, 48, 6

            # in case of 3d sampling, we need to add the weights
            if ray_bundle.nears is not None:
                weights = weights + 1.0

            print("desnity mask shape", density_mask.shape, " semantics: ", field_outputs_dict["semantics"].shape)
            semantics[density_mask] = field_outputs_dict["semantics"]  # 1, num_dense_samples, 6
            weights_all[density_mask] = weights

            semantics[~density_mask] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)
            weights_all[~density_mask] = 0.01

            semantics = self.renderer_semantics(semantics, weights=weights_all)
            outputs["semantics"] = semantics  # N, num_classes

            # semantics colormaps
            semantic_labels = torch.argmax(torch.nn.functional.softmax(semantics, dim=-1), dim=-1)

            self.semantics.colors = self.semantics.colors.to(self.device)
            semantics_colormap = self.semantics.colors[semantic_labels].to(self.device)
            outputs["semantics_colormap"] = semantics_colormap
            outputs["rgb"] = semantics_colormap

            # print the count of the different labels
            # CONSOLE.print("Label counts:", torch.bincount(semantic_labels.flatten()))
        if "density" in self.config.mode:
            density = torch.empty((*density_mask.shape, 1), device=self.device)
            density[density_mask] = field_outputs_dict["density"]
            # density[~density_mask] = torch.nn.functional.relu(self.learned_low_density_value_density)
            density[~density_mask] = 0.0

            # make it predict logarithmic density instead
            density = torch.exp(density) - 1.0

            weights = original_fields_outputs["ray_samples"].get_weights(density)

            rgb = self.renderer_rgb(original_fields_outputs[FieldHeadNames.RGB], weights=weights)
            outputs["rgb"] = rgb
            outputs["density_pred"] = density

            # filter out high density values > 4000 and set them to maximum
            density_gt = original_fields_outputs[FieldHeadNames.DENSITY]
            density_gt[density_gt > self.config.density_cutoff] = self.config.density_cutoff
            outputs["density_gt"] = density_gt
            # with p =0.1 log histograms
            if random.random() < 0.01:
                if wandb.run is not None:
                    wandb.log(
                        {
                            "density/pred": get_wandb_histogram(density),
                            "density/gt": get_wandb_histogram(density_gt),
                        },
                        step=wandb.run.step,
                    )
        if "normals" in self.config.mode:
            time13 = time.time()
            
            assert self.config.sampler.surface_sampling, f"Surface sampling is required for normal prediction but is {self.config.sampler.surface_sampling}" 

            # normalize to unit vecctors
            field_outputs = torch.nn.functional.normalize(field_outputs_dict["normals"], dim=-1)

            # rotate outputs back            
            rot_mat = transform_batch["rot_mat"].transpose(1, 2)
            field_outputs = torch.matmul(field_outputs, rot_mat)
            transform_batch["normals"] = torch.matmul(transform_batch["normals"], rot_mat)

            outputs["normals_pred"] = field_outputs
            outputs["normals_gt"] = transform_batch["normals"]
            
            # for visualization
            normals_all = torch.empty((*density_mask.shape, 3), device=self.device)
            normals_gt = torch.empty((*density_mask.shape, 3), device=self.device)
            weights_all = torch.zeros((*density_mask.shape, 1), device=self.device)  # 64, 48, 6

            normals_all[density_mask] = field_outputs
            normals_gt[density_mask] = transform_batch["normals"]
            weights_all[density_mask] = weights

            normals_all[~density_mask] = torch.nn.functional.normalize(torch.nn.functional.tanh(self.learned_low_density_value_normals), dim=-1)
            normals_gt[~density_mask] = torch.zeros((3), device=normals_gt.device)
            weights_all[~density_mask] = 0.00001
            
            time14 = time.time()
            outputs["normals_all_pred"] = torch.nn.functional.normalize(self.renderer_normals(normals_all, weights=weights_all), dim=-1)
            outputs["normals_all_gt"] = torch.nn.functional.normalize(self.renderer_normals(normals_gt, weights=weights_all), dim=-1)
            time15 = time.time()
            CONSOLE.print("Forward - data post processing time: ", time15 - time14)
            CONSOLE.print("Forward - render: ", time14 - time13)
        
        if self.config.visualize_semantic_pc:
            assert self.config.mode == "semantics"
            semantic_labels = torch.argmax(outputs["semantics"][density_mask.squeeze()].unsqueeze(0), dim=-1)
            visualize_point_batch(transform_batch["points_xyz"], classes=semantic_labels)
            visualize_point_batch(transform_batch["points_xyz"], classes=torch.argmax(field_outputs, dim=-1))
        
        outputs["density_mask"] = density_mask

        return outputs

    def forward(self, ray_bundle: RayBundle, batch: Union[Dict[str, Any], None] = None) -> Dict[str, torch.Tensor]:
        return self.get_outputs(ray_bundle, batch)

    def enrich_dict_with_model(self, d: dict, model_idx: int) -> dict:
        keys = list(d.keys())

        for key in keys:
            d[key + "_" + str(model_idx)] = d[key]
        d["model_idx"] = model_idx

        return d

    def get_metrics_dict(self, outputs, batch: Dict[str, Any]):
        metrics_dict = {}
        if "eval_model_idx" in batch:
            metrics_dict["eval_model_idx"] = batch["eval_model_idx"]

        if "rgb" in self.config.mode:
            with torch.no_grad():
                image = batch["image"].to(self.device)
                metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
                metrics_dict["psnr_" + str(batch["model_idx"])] = metrics_dict["psnr"]

                metrics_dict["mse"] = F.mse_loss(outputs["rgb"], image)
                metrics_dict["mse_" + str(batch["model_idx"])] = metrics_dict["mse"]

                metrics_dict["mae"] = F.l1_loss(outputs["rgb"], image)
                metrics_dict["mae_" + str(batch["model_idx"])] = metrics_dict["mae"]

        if "semantics" in self.config.mode:
            semantics = batch["semantics"][..., 0].long().to(self.device)
            with torch.no_grad():
                confusion = self.confusion_non_normalized(torch.argmax(outputs["semantics"], dim=-1), semantics).detach().cpu().numpy()
                miou, per_class_iou = compute_mIoU(confusion)
                total_acc, acc_per_class = calculate_accuracy(confusion)
                # mIoU
                metrics_dict["miou"] = miou
                metrics_dict["acc"] = total_acc
                for i, iou in enumerate(per_class_iou):
                    metrics_dict["iou_" + self.semantics.classes[i]] = iou
                    
                for i, acc in enumerate(acc_per_class):
                    metrics_dict["acc_" + self.semantics.classes[i]] = acc

                metrics_dict["miou_" + str(batch["model_idx"])] = metrics_dict["miou"]
        if "density" in self.config.mode:
            image = batch["image"].to(self.device)

            with torch.no_grad():
                metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
                metrics_dict["psnr_" + str(batch["model_idx"])] = metrics_dict["psnr"]

                metrics_dict["mse"] = F.mse_loss(outputs["rgb"], image)
                metrics_dict["mse_" + str(batch["model_idx"])] = metrics_dict["mse"]

                metrics_dict["mae"] = F.l1_loss(outputs["rgb"], image)
                metrics_dict["mae_" + str(batch["model_idx"])] = metrics_dict["mae"]

                metrics_dict["density_mse"] = F.mse_loss(outputs["density_gt"], outputs["density_pred"])
                metrics_dict["density_mse" + str(batch["model_idx"])] = metrics_dict["density_mse"]

                metrics_dict["density_mae"] = F.l1_loss(outputs["density_gt"], outputs["density_pred"])
                metrics_dict["density_mae_" + str(batch["model_idx"])] = metrics_dict["density_mae"]
        if "normals" in self.config.mode:
            normals_pred = outputs["normals_pred"]
            normals_gt = outputs["normals_gt"]
            metrics_dict["dot"] = (1.0 - torch.sum(normals_gt * normals_pred, dim=-1)).mean(dim=-1)
            
            if "normal_image" in batch:
                density_mask = outputs["density_mask"].squeeze()
                normals_all_pred = outputs["normals_all_pred"][density_mask]
                normals_all_analytic = outputs["normals_all_gt"][density_mask]
                normals_all_gt = torch.nn.functional.normalize(batch["normal_image"][density_mask] * 2.0 - 1.0)
                
                metrics_dict["gt_analytic_dot"] = torch.sum(normals_all_gt * normals_all_analytic, dim=-1).mean(dim=-1)
                metrics_dict["gt_analytic_dot_" + str(batch["model_idx"])] = metrics_dict["gt_analytic_dot"]
                
                metrics_dict["gt_pred_dot"] = torch.sum(normals_all_pred * normals_all_analytic, dim=-1).mean(dim=-1)
                metrics_dict["gt_pred_dot_" + str(batch["model_idx"])] = metrics_dict["gt_pred_dot"]
            
        return metrics_dict

    def get_loss_dict(self, outputs, batch: Dict[str, Any], metrics_dict=dict):
        loss_dict = {}
        if "rgb" in self.config.mode:
            image = batch["image"].to(self.device)[outputs["density_mask"].squeeze()]

            model_output = outputs["rgb"][outputs["density_mask"].squeeze()]
            if self.config.rgb_prediction == "integration":
                loss_dict["rgb_loss"] = self.rgb_loss(image, model_output)
            elif self.config.rgb_prediction == "direct":
                loss_dict["rgb_loss"] = self.rgb_loss(outputs["rgb_pred"], outputs["rgb_gt"])
        if "semantics" in self.config.mode:
            density_mask_accu = torch.any(outputs["density_mask"], dim=-1)
            pred = outputs["semantics"][density_mask_accu.squeeze()]
            gt = batch["semantics"][..., 0].long().to(self.device)[density_mask_accu.squeeze()]
            loss_dict["semantics_loss"] = self.cross_entropy_loss(pred, gt)
        if "density" in self.config.mode:
            if self.config.density_prediction == "direct":
                loss_dict["density"] = self.density_loss(outputs["density_gt"], outputs["density_pred"])
            elif self.config.density_prediction == "integration":
                image = batch["image"].to(self.device)
                model_output = outputs["rgb"]
                loss_dict["density"] = self.density_loss(image, model_output)
            else:
                raise ValueError("Unknown density prediction mode: " + self.config.density_prediction)
        if "normals" in self.config.mode:
            loss_dict["dot"] = metrics_dict["dot"]
            
        if self.config.proximity_loss:
            loss_dict["proximity_loss"] = torch.nn.functional.mse_loss(outputs["pointcloud_pred"], outputs["pointcloud_pred_noise"]) * self.config.proximity_loss_mult
        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: Union[RayBundle, List[RayBundle]], batch: Union[Dict[str, Any], None] = None
    ) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
            :param batch: additional information of the batch here it includes at least the model
        """
        if isinstance(camera_ray_bundle, list):
            images = len(camera_ray_bundle[1:])
            camera_ray_bundle_query = camera_ray_bundle[0]
            camera_ray_bundle_np = stack_ray_bundles(camera_ray_bundle[1:])
            length = len(camera_ray_bundle_np)
            camera_ray_bundle_np = camera_ray_bundle_np.flatten()[torch.randperm(length)][:batch["neural_point_cloud_size"]]
            batch["ray_bundle_np"] = camera_ray_bundle_np
            image_height, image_width = camera_ray_bundle_query.shape
            use_all_pixels = True
        else:
            assert False, "Not longer supported"
            use_all_pixels = True
            image_height, image_width = camera_ray_bundle.origins.shape[:2]

        def batch_randomly(max_length, batch_size):
            indices = torch.randperm(max_length)
            reverse_indices = torch.argsort(indices)
            return indices, reverse_indices

        def batch_evenly(max_length, batch_size):
            indices = torch.arange(max_length)
            final_indices = []
            reverse_indices = torch.zeros(max_length, dtype=torch.long)

            step_size = max_length // batch_size + 1
            running_length = 0
            for i in range(step_size):
                ind = indices[i::step_size]
                length_ind = ind.size()[0]
                final_indices.append(ind)
                reverse_indices[ind] = torch.arange(running_length, running_length + length_ind, dtype=torch.long)
                running_length += length_ind

            return torch.cat(final_indices, dim=0), reverse_indices

        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        num_rays = len(camera_ray_bundle_query)
        outputs_lists = defaultdict(list)
        if self.config.space_partitioning == "evenly":
            ray_order, reversed_ray_order = batch_evenly(num_rays, num_rays_per_chunk)
        elif self.config.space_partitioning == "random":
            ray_order, reversed_ray_order = batch_randomly(num_rays, num_rays_per_chunk)
        else:
            ray_order = []
            reversed_ray_order = None

        # get permuted ind
        for i in range(0, num_rays, num_rays_per_chunk):
            if self.config.space_partitioning != "row_wise":
                indices = ray_order[i : i + num_rays_per_chunk]
                ray_bundle = camera_ray_bundle_query.flatten()[indices]
            else:
                start_idx = i
                end_idx = i + num_rays_per_chunk
                ray_bundle = camera_ray_bundle_query.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle, batch=batch)
            batch["neural_points"] = outputs["neural_points"]
            batch["neural_features"] = outputs["neural_features"]

            for output_name, output in outputs.items():  # type: ignore
                outputs_lists[output_name].append(output)
        outputs = {
            "neural_points": batch["neural_points"],
            "neural_features": batch["neural_features"],
        }
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue
            if self.config.space_partitioning != "row_wise":
                if  output_name == "normals_pred" or output_name == "normals_gt" or output_name == "pointcloud_pred" or output_name == "pointcloud_pred_noise" \
                    or output_name == "neural_points" or output_name == "neural_features":
                    continue
                # TODO maybe fix later
                    unordered_output_tensor = torch.cat(outputs_list, dim=1).squeeze(0)
                else:
                    unordered_output_tensor = torch.cat(outputs_list)
                ordered_output_tensor = unordered_output_tensor[reversed_ray_order]
            else:
                ordered_output_tensor = torch.cat(outputs_list)  # type: ignore

            if not use_all_pixels:
                ordered_output_tensor = ordered_output_tensor[: image_height * image_width]
            outputs[output_name] = ordered_output_tensor.view(image_height, image_width, -1)  # type: ignore
        return outputs

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        images_dict = {}
        metrics_dict = {}

        if "rgb" in self.config.mode or "density" in self.config.mode:
            image = batch["image"].to(self.device)
            rgb = outputs["rgb"]
            combined_rgb = torch.cat([image, rgb], dim=1)
            images_dict["imgs_rgb"] = combined_rgb

            # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
            image = torch.moveaxis(image, -1, 0)[None, ...]
            rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

            with torch.no_grad():
                psnr = self.psnr(image, rgb)
                metrics_dict["psnr"] = float(psnr.item())
                # metrics_dict["psnr_" + str(batch["model_idx"])] = metrics_dict["psnr"]

                metrics_dict["mse"] = F.mse_loss(image, rgb).item()
                # metrics_dict["mse_" + str(batch["model_idx"])] = metrics_dict["mse"]

                metrics_dict["mae"] = F.l1_loss(image, rgb).item()
                # metrics_dict["mae_" + str(batch["model_idx"])] = metrics_dict["mae"]

                metrics_dict["ssim"] = self.ssim(image, rgb).item()
                # metrics_dict["ssim_" + str(batch["model_idx"])] = metrics_dict["ssim"]

                if self.config.mode == "density":
                    metrics_dict["density_mse"] = F.mse_loss(outputs["density_gt"], outputs["density_pred"]).item()

                    metrics_dict["density_mae"] = F.l1_loss(outputs["density_gt"], outputs["density_pred"]).item()

        if "semantics" in self.config.mode:
            semantics_colormap_gt = self.semantics.colors[batch["semantics"].squeeze(-1).to("cpu")].to(self.device)
            semantics_colormap = outputs["semantics_colormap"]
            combined_semantics = torch.cat([semantics_colormap_gt, semantics_colormap], dim=1)
            images_dict["imgs_semantics"] = combined_semantics
            images_dict["semantics_colormap"] = outputs["semantics_colormap"]
            images_dict["rgb_image"] = batch["image"]
            
            # compute uncertainty entropy
            def compute_entropy(tensor):
                # Reshape tensor to [B, 256*256, C]
                tensor = tensor.view(tensor.size(0), -1, tensor.size(-1))
                
                # Compute softmax probabilities
                probabilities = F.softmax(tensor, dim=-1)
                
                # Compute entropy
                entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-10), dim=-1)
                
                # Normalize entropy to [0, 1]
                max_entropy = torch.log2(torch.tensor(tensor.size(-1)).float())
                normalized_entropy = entropy / max_entropy
                
                return normalized_entropy.unsqueeze(-1)

            images_dict["entropy"] = compute_entropy(outputs["semantics"])
            
            density_mask_accu = torch.any(outputs["density_mask"], dim=-1).reshape(-1)
            outs = outputs["semantics"].reshape(-1, outputs["semantics"].shape[-1]).to(self.device)
            gt = batch["semantics"][..., 0].long().reshape(-1)
            pred = torch.argmax(outs, dim=-1)
            confusion_non_normalized = self.confusion_non_normalized(pred, gt).detach().cpu().numpy()
            miou, per_class_iou = compute_mIoU(confusion_non_normalized)
            total_acc, acc_per_class = calculate_accuracy(confusion_non_normalized)

            # mIoU
            metrics_dict["miou"] = miou
            metrics_dict["accuracy"] = total_acc
            metrics_dict["confusion_unnormalized"] = confusion_non_normalized
            for i, iou in enumerate(per_class_iou):
                metrics_dict["iou_" + self.semantics.classes[i]] = iou

            for i, acc in enumerate(acc_per_class):
                metrics_dict["acc_" + self.semantics.classes[i]] = acc

            confusion = self.confusion(pred, gt).detach().cpu().numpy()
            if self.config.plot_confusion:
                fig = ff.create_annotated_heatmap(confusion, x=self.semantics.classes, y=self.semantics.classes)
                fig.update_layout(title="Confusion matrix", xaxis_title="Predicted", yaxis_title="Actual")
                fig.show()

            if self.config.log_confusion_to_wandb and wandb.run is not None:
                fig = ff.create_annotated_heatmap(confusion, x=self.semantics.classes, y=self.semantics.classes)
                fig.update_layout(title="Confusion matrix", xaxis_title="Predicted", yaxis_title="Actual")
                wandb.log(
                    {"confusion_matrix": fig},
                    step=wandb.run.step,
                )
        if "normals" in self.config.mode:
            if "normals_all_pred" in outputs:
                images_dict["normals_all_pred"] = (outputs["normals_all_pred"] + 1.0) / 2.0
            if "normals_all_gt" in outputs:
                images_dict["normals_all_gt"] = (outputs["normals_all_gt"] + 1.0) / 2.0
            
            if "normals_all_pred" in outputs and "normals_all_gt" in outputs:
                normals_pred = outputs["normals_all_pred"][outputs["density_mask"].squeeze()]
                normals_analytic = outputs["normals_all_gt"][outputs["density_mask"].squeeze()]
                metrics_dict["analytic_pred_dot"] = torch.sum(normals_analytic * normals_pred, dim=-1).mean(dim=-1).item()
            
            if "normal_image" in batch:
                images_dict["normals_gt"] = batch["normal_image"]
                density_mask = outputs["density_mask"].squeeze()
                normals_all_pred =outputs["normals_all_pred"][density_mask]
                normals_all_analytic = outputs["normals_all_gt"][density_mask]
                normals_all_gt = torch.nn.functional.normalize(batch["normal_image"][density_mask] * 2.0 - 1.0)
                
                metrics_dict["gt_analytic_dot"] = torch.sum(normals_all_gt * normals_all_analytic, dim=-1).mean(dim=-1).item()
                
                metrics_dict["gt_pred_dot"] = torch.sum(normals_all_gt * normals_all_pred, dim=-1).mean(dim=-1).item()

            
            normal_images = [images_dict[key] for key in ["normals_all_pred", "normals_all_gt"] if key in images_dict]
            if "normal_image" in batch:
                normal_images.append(batch["normal_image"])

            images_dict["imgs_normals"] = torch.cat(normal_images, dim=1)
        # plotly show image
        if self.config.debug_show_image:
            fig = px.imshow(images_dict["img"].cpu().numpy())
            fig.show()
        
        if "density_mask" in outputs:
            images_dict["sample_mask"] = torch.mean(outputs["density_mask"].float(), dim=-1, keepdim=True)

        return metrics_dict, images_dict

    def set_model(self, model: Model):
        """Sets the fallback model for inference of the Neural Semantic Field.
        It is a nerf model which can be queried to obtain points.

        Args:
            model (Model): The fallback nerf model
        """
        # set the model to not require a gradient
        for param in model.parameters():
            param.requires_grad = False
        self.fallback_model = model

    @profiler.time_function
    def get_model(self, batch: Union[Dict[str, Any], None]) -> Model:
        """Gets the fallback model for inference of the Neural Semantic Field.
        It is a nerf model which can be queried to obtain points.

        Returns:
            Model: The fallback nerf model
        """
        if batch is None or "model" not in batch:
            assert self.fallback_model is not None
            CONSOLE.print("Using fallback model for inference")
            model = self.fallback_model
        else:
            # CONSOLE.print("Using batch model for inference")
            model = batch["model"]
        model.eval()
        return model

    def batching(self, ray_samples: RaySamples, field_outputs: dict):
        if self.config.batching_mode != "off":
            if self.config.batching_mode == "sequential":
                # given the features and the density mask batch them up sequentially such that each batch is same size.
                field_outputs, masking, ids_shuffle, ids_restore, points_pad, directions_pad = sequential_batching(
                    ray_samples, field_outputs, self.config.batch_size
                )
            elif self.config.batching_mode == "random":
                field_outputs, masking, ids_shuffle, ids_restore, points_pad, directions_pad = random_batching(
                    ray_samples, field_outputs, self.config.batch_size
                )
            elif self.config.batching_mode == "sliced":
                field_outputs, masking, ids_shuffle, ids_restore, points_pad, directions_pad = spatial_sliced_batching(
                    ray_samples, field_outputs, self.config.batch_size, self.scene_box.aabb
                )
            else:
                raise ValueError(f"Unknown batching mode {self.config.batching_mode}")

            # sorting by ids rearangement. Not necessary for sequential batching
            for k, v in field_outputs.items():
                field_outputs[k] = v[ids_shuffle, :]

            points_pad = points_pad[ids_shuffle, :]
            directions_pad = directions_pad[ids_shuffle, :]
            mask = masking[ids_shuffle]

            # batching
            for k, v in field_outputs.items():
                field_outputs[k] = v.reshape(-1, self.config.batch_size, v.shape[-1])
            mask = mask.reshape(-1, self.config.batch_size)

            transform_batch = {
                "ids_shuffle": ids_shuffle,
                "ids_restore": ids_restore,
                "src_key_padding_mask": mask,
                "points_xyz": points_pad.reshape(*mask.shape, 3),
                "directions": directions_pad.reshape(*mask.shape, 3),
            }
        else:
            W = None
            transform_batch = {
                "ids_shuffle": None,
                "ids_restore": None,
                "src_key_padding_mask": None,
                "points_xyz": ray_samples.frustums.get_positions().unsqueeze(0),
                "directions": ray_samples.frustums.directions.unsqueeze(0),
            }

            for k, v in field_outputs.items():
                field_outputs[k] = v.unsqueeze(0)

        return field_outputs, transform_batch
