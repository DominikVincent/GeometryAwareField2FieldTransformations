import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple, Type, Union, cast

import numpy as np
import torch
import torch_points_kernels as tp
from model.stratified_transformer import Stratified
from pointnet.models.pointnet2_sem_seg import get_model
from rich.console import Console
from sklearn.linear_model import RANSACRegressor
from torch import nn
from torch.distributions import uniform
from torch.nn import Parameter
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import NeRFEncoding, RFFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.nerfacto_field import get_normalized_directions
from nerfstudio.models.base_model import Model
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.utils.nesf_utils import (
    log_points_to_wandb,
    visualize_point_batch,
    visualize_points,
    visualize_rgb_point_cloud,
    visualize_rgb_point_cloud_masked
)

CONSOLE = Console(width=120)


@dataclass
class SceneSamplerConfig(InstantiateConfig):
    """target class to instantiate"""

    _target: Type = field(default_factory=lambda: SceneSampler)

    samples_per_ray: int = 16
    """How many samples per ray to take"""
    surface_sampling: bool = True
    """Sample only the surface or also the volume"""
    surface_threshold: float = 0.5
    """At what accumulated weight percentage along a ray a surface is considered"""
    density_threshold: float = 0.7
    """The density threshold for which to not use the points for training. Points below will not be used"""
    z_value_threshold: float = -1
    """What is the minimum z value a point has to have"""
    xy_distance_threshold: float = 0.7
    """The maximal distance a point can have to z axis to be considered"""
    max_points: int = 80000
    """The maximum number of points to use in one scene. If more are available after filtering, they will be randomly sampled"""
    get_normals: bool = False
    ground_removal_mode: Literal["ransac", "min", "none"] = "none"
    """Get normals for the samples"""
    ground_points_count: int = 500
    """How many points to sample for the ground"""
    ground_tolerance: float = 0.0075
    """The distance a point has to have to the min z value to be considered non ground"""



class SceneSampler:
    """_summary_ A class which samples a scene given a ray bundle.
    It will filter out points/ray_samples and batch up the scene.
    """

    def __init__(self, config: SceneSamplerConfig):
        self.config = config

        # maps scene to plane parameters (a, b, c, d)
        self.plane_cache = {}

    def sample_scene(self, ray_bundle: RayBundle, model: Model, model_idx: int) -> Tuple[RaySamples, torch.Tensor, dict, torch.Tensor, dict]:
        """_summary_
        Samples the model for a given ray bundle. Filters and batches points.

        Args:
            ray_bundle (_type_): A ray bundle. Might be from different cameras.
            model (_type_): A nerf model. Currently NerfactoModel is supported..

        Returns:
        - ray_samples (_type_): The ray samples for the scene which should be used. (N_dense)
        - weights (_type_): The weights for the ray samples which are used. (N_dense x 1)
        - field_outputs (_type_): The field outputs for the ray samples which are used. (N_dense x dim)
        - final_mask (_type_): The mask for the ray samples which are used. Is the shape of the original ray_samples. (N_rays x N_samples)
        - original_fields_outputs (_type_): The original field outputs for the ray samples. (N_rays x N_samples x dim) N_samples = 1 in case of surface sampling.

        Raises:
            NotImplementedError: _description_
        """
        model.eval()
        time1 = time.time()
        if isinstance(model, NerfactoModel):
            model = cast(NerfactoModel, model)
            with torch.no_grad():
                if model.collider is not None:
                    ray_bundle = model.collider(ray_bundle)

                model.proposal_sampler.num_nerf_samples_per_ray = self.config.samples_per_ray
                ray_samples, _, _ = model.proposal_sampler(ray_bundle, density_fns=model.density_fns)
                field_outputs = model.field(ray_samples, compute_normals=self.config.get_normals)
                field_outputs[FieldHeadNames.DENSITY] = field_outputs[FieldHeadNames.DENSITY].detach()  # type: ignore
                # reset to avoid memory wastage
                model.field._density_before_activation = None
                model.field._sample_locations = None
                weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        else:
            raise NotImplementedError("Only NerfactoModel is supported for now")

        # visualize_rgb_point_cloud(ray_samples.frustums.get_positions(), field_outputs[FieldHeadNames.RGB])

        time2 = time.time()
        if self.config.surface_sampling:
            ray_samples, weights, field_outputs = self.surface_sampling(ray_samples, weights, field_outputs)

        time3 = time.time()
        original_fields_outputs = {}
        original_fields_outputs[FieldHeadNames.DENSITY] = field_outputs[FieldHeadNames.DENSITY].detach().clone()
        original_fields_outputs[FieldHeadNames.RGB] = field_outputs[FieldHeadNames.RGB].detach().clone()
        time4 = time.time()
        if FieldHeadNames.PRED_NORMALS in field_outputs:
            original_fields_outputs[FieldHeadNames.PRED_NORMALS] = field_outputs[FieldHeadNames.PRED_NORMALS].detach().clone()
        else:
            if FieldHeadNames.NORMALS in field_outputs:
                original_fields_outputs[FieldHeadNames.NORMALS] = field_outputs[FieldHeadNames.NORMALS].detach().clone()

        original_fields_outputs["ray_samples"] = ray_samples

        density_mask = self.get_density_mask(field_outputs)
        time5 = time.time()
        pos_mask = self.get_pos_mask(ray_samples)
        time6 = time.time()

        total_mask = density_mask & pos_mask


        # only compute ground points from filter points
        if self.config.ground_removal_mode == "ransac":
            non_ground_mask = self.get_non_ground_mask_ground_plane_fitting(ray_samples, total_mask, model_idx)
        elif self.config.ground_removal_mode == "min":
            non_ground_mask = self.get_non_ground_mask(ray_samples, total_mask)
        elif self.config.ground_removal_mode == "none":
            non_ground_mask = torch.ones_like(total_mask)
        else:
            raise NotImplementedError("Only ransac and min are supported for now")

        time7 = time.time()
        total_mask = total_mask & non_ground_mask

        final_mask = self.get_limit_mask(total_mask)
        time8 = time.time()
        ray_samples, weights, field_outputs = self.apply_mask(ray_samples, weights, field_outputs, final_mask)
        time9 = time.time()

        # visualize_point_batch(ray_samples.frustums.get_positions())
        # CONSOLE.print(f"Sampler: query nerf: {time2-time1}")
        # CONSOLE.print(f"Sampler: surface sampling: {time3-time2}")
        # CONSOLE.print(f"Sampler: copy field outputs: {time4-time3}")
        # CONSOLE.print(f"Sampler: density mask: {time5-time4}")
        # CONSOLE.print(f"Sampler: pos mask: {time6-time5}")
        # CONSOLE.print(f"Sampler: non ground mask: {time7-time6}")
        # CONSOLE.print(f"Sampler: limit mask: {time8-time7}")
        # CONSOLE.print(f"Sampler: apply mask: {time9-time8}")
        return ray_samples, weights, field_outputs, final_mask, original_fields_outputs

    def surface_sampling(
        self, ray_samples: RaySamples, weights: TensorType["N_rays", "N_samples", 1], field_outputs: dict
    ):
        """Samples only surface samples along the ray

        Args:
            ray_samples (_type_): N_rays x N_samples
            weights (_type_): N_rays x N_samples x 1
            field_outputs (_type_): N_rays x N_samples x dim

        Returns:
            ray_samples (_type_): N_rays x 1
            weights (_type_): N_rays x 1 x 1
            field_outputs (_type_): N_rays x 1 x dim
        """
        cumulative_weights = torch.cumsum(weights[..., 0], dim=-1)  # [..., num_samples]
        split = torch.ones((*weights.shape[:-2], 1), device=weights.device) * self.config.surface_threshold  # [..., 1]
        median_index = torch.searchsorted(cumulative_weights, split, side="left")  # [..., 1]
        median_index = torch.clamp(median_index, 0, ray_samples.shape[-1] - 1)  # [..., 1]

        for k, v in field_outputs.items():
            field_outputs[k] = v[torch.arange(median_index.shape[0]), median_index.squeeze(), ...].unsqueeze(1)

        # field_outputs[FieldHeadNames.RGB] = field_outputs[FieldHeadNames.RGB][
        #     torch.arange(median_index.shape[0]), median_index.squeeze(), ...
        # ].unsqueeze(1)
        # field_outputs[FieldHeadNames.DENSITY] = field_outputs[FieldHeadNames.DENSITY][
        #     torch.arange(median_index.shape[0]), median_index.squeeze(), ...
        # ].unsqueeze(1)
        weights = weights[torch.arange(median_index.shape[0]), median_index.squeeze(), ...].unsqueeze(1)
        ray_samples = ray_samples[torch.arange(median_index.shape[0]), median_index.squeeze()].unsqueeze(1)

        return ray_samples, weights, field_outputs

    def get_density_mask(self, field_outputs) -> TensorType["N_rays", "N_samples"]:
        # true for points to keep
        density = field_outputs[FieldHeadNames.DENSITY]  # 64, 48, 1 (rays, samples, 1)
        density_mask = density > self.config.density_threshold  # 64, 48
        density_mask = density_mask.squeeze(-1)
        return density_mask

    def get_pos_mask(self, ray_samples) -> TensorType["N_rays", "N_samples"]:
        # true for points to keep
        points = ray_samples.frustums.get_positions()

        # visualize_point_batch(points.cpu().view(-1, 3))

        points_dense_mask = (points[..., 2] > self.config.z_value_threshold) & (
            torch.norm(points[..., :2], dim=-1) <= self.config.xy_distance_threshold
        )
        # visualize_point_batch(points[points_dense_mask].cpu().view(-1, 3))

        # print(points_dense_mask)
        return points_dense_mask

    def get_non_ground_mask(self, ray_samples, mask: TensorType) -> TensorType["N_rays", "N_samples"]:
        # returns true for non ground points
        points = ray_samples.frustums.get_positions()

        filtered_points = points[mask]

        sorted_z = torch.sort(filtered_points[..., 2].flatten())[0]
        ground_z = sorted_z[:self.config.ground_points_count].median()

        distances = points[..., 2] - ground_z

        # Create a mask for all points with a distance greater than the tolerance
        mask = distances > self.config.ground_tolerance

        return mask

    def get_non_ground_mask_ground_plane_fitting(self, ray_samples, mask: TensorType, scene_idx: int) -> TensorType["N_rays", "N_samples"]:
        # returns true for non ground points
        time_start = time.time()
        points = ray_samples.frustums.get_positions()


        if scene_idx in self.plane_cache:
            a, b, c, d = self.plane_cache[scene_idx]
        else:
            filtered_points = points[mask]
            # visualize_point_batch(points.cpu().unsqueeze(0))

            # find lowest z point
            sorted_z = torch.sort(filtered_points[..., 2].flatten())[0]
            ground_z = sorted_z[:self.config.ground_points_count].median()

            filtered_points = filtered_points[filtered_points[..., 2] < ground_z + 2 * self.config.ground_tolerance]

            # visualize_point_batch(filtered_points.cpu().unsqueeze(0))

            model = RANSACRegressor(max_trials=100)
            model.fit(filtered_points[..., :2].cpu(), filtered_points[..., 2].cpu())

            # Extract the parameters of the fitted plane
            # TODO check that c is always -1
            a, b, c, d = model.estimator_.coef_[0], model.estimator_.coef_[1], -1, model.estimator_.intercept_
            self.plane_cache[scene_idx] = (a, b, c, d)

        if c < 0:
            a, b, c, d = -a, -b, -c, -d

        # Calculate the distance of each point from the fitted plane
        distances = (a * points[..., 0] + b * points[..., 1] + c * points[..., 2] + d) / np.sqrt(a**2 + b**2 + c**2)

        # Create a mask for all points with a distance greater than the threshold
        mask = distances > self.config.ground_tolerance

        time_end = time.time()
        print("Ground plane fitting took: ", time_end - time_start, " seconds")
        print("Mask", mask)
        return mask

    def get_limit_mask(self, mask):
        num_true = int(torch.sum(mask).item())

        # If there are more than k true values, randomly select which ones to keep
        if num_true > self.config.max_points:
            CONSOLE.print(f"[yellow]Limiting number of points from {num_true} to {self.config.max_points}")
            true_indices = torch.nonzero(mask, as_tuple=True)
            num_true_values = true_indices[0].size(0)

            # Randomly select k of the true indices
            selected_indices = torch.randperm(num_true_values)[: self.config.max_points]

            # Create a new mask with only the selected true values
            new_mask = torch.zeros_like(mask)
            new_mask[true_indices[0][selected_indices], true_indices[1][selected_indices]] = 1
            return new_mask
        else:
            return mask

    def apply_mask(
        self,
        ray_samples: RaySamples,
        weights: TensorType["N_rays", "N_samples", 1],
        field_outputs,
        mask: TensorType["N_rays", "N_samples"],
    ) -> Tuple[RaySamples, TensorType["N_dense", 1], dict]:
        """Applies the mask to the ray samples and field outputs

        Returns:
            ray_samples: Will be N_dense
            field_outputs: Will be N_dense x dim
        """
        for k, v in field_outputs.items():
            field_outputs[k] = v[mask]

        ray_samples = ray_samples[mask]
        weights = weights[mask]

        # all but density_mask should have the reduced size
        return ray_samples, weights, field_outputs

    def clear_ground_cache(self):
        self.plane_cache = {}

@dataclass
class MaskerConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Masker)

    mask_ratio: float = 0.75
    """The number of points which should be masked approximatly"""

    mode: Literal["random", "patch", "patch_fp"] = "random"
    """The mode of masking to use"""

    pos_encoder: Literal["sin", "rff"] = "sin"
    """what kind of feature encoded should be used?"""

    num_patches: int = 25
    """How many centroids should be used for the patch mode. Patches will evolve around centroids"""

    visualize_masking: bool = False

class Masker(nn.Module):
    def __init__(self, config: MaskerConfig, output_size: int):
        super().__init__()

        self.config = config

        mask_token_value = torch.empty(1, 1, output_size)
        torch.nn.init.kaiming_normal_(mask_token_value)
        self.mask_token = torch.nn.Parameter(mask_token_value, requires_grad=True)

        if self.config.pos_encoder == "sin":
            self.pos_encoder = NeRFEncoding(
                in_dim=3, num_frequencies=7, min_freq_exp=0.0, max_freq_exp=7.0, include_input=True
            )
        elif self.config.pos_encoder == "rff":
            self.pos_encoder = RFFEncoding(in_dim=3, num_frequencies=8, scale=10)
        else:
            raise ValueError(f"Unknown pos encoder {self.config.pos_encoder}")
        assert self.pos_encoder.get_out_dim() <= output_size

    def forward(self, x: torch.Tensor, batch: dict):
        self.mask(x, batch)

    def mask(self, x: torch.Tensor, batch: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Mask a certain amount of points in the batch for the encoder. It will filter out the mask points"""
        if self.config.mask_ratio == 0:
            return x, torch.empty((1,)), torch.empty((1,)), {}

        if self.config.mode == "random":
            ids_keep, ids_mask, ids_restore =  self.random_mask(x, batch)
        elif self.config.mode == "patch" or self.config.mode == "patch_fp":
            ids_keep, ids_mask, ids_restore =  self.patch_mask(x, batch)
        else:
            raise ValueError(f"Unknown masking mode {self.config.mode}")

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.config.mask_ratio))

        batch["points_xyz_all"] = batch["points_xyz"]
        batch["points_xyz"] = torch.gather(batch["points_xyz"], dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 3))
        rgb_keep = torch.gather(batch["rgb"], dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 3))
        batch["points_xyz_masked"] = torch.gather(batch["points_xyz_all"], dim=1, index=ids_mask.unsqueeze(-1).repeat(1, 1, 3))
        rgb_masked = torch.gather(batch["rgb"], dim=1, index=ids_mask.unsqueeze(-1).repeat(1, 1, 3))
        if "src_key_padding_mask" in batch and batch["src_key_padding_mask"] is not None:
            batch["src_key_padding_mask_orig"] = batch["src_key_padding_mask"]
            batch["src_key_padding_mask"] = torch.gather(batch["src_key_padding_mask"], dim=1, index=ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        if self.config.visualize_masking:
            # visualize_rgb_point_cloud_masked(batch["points_xyz"], rgb_keep, batch["points_xyz_masked"])
            points_stack = torch.cat([batch["points_xyz"], batch["points_xyz_masked"]], dim=1)
            labels = torch.cat([torch.ones((N, len_keep)), torch.zeros((N, L - len_keep))], dim=1).long()
            visualize_point_batch(points_stack, classes=labels)
            input("Press Enter to continue...")
        return x_masked, mask, ids_restore, batch

    def unmask(self, x: torch.Tensor, batch: dict, ids_restore: torch.Tensor):
        """Unmask the points in the batch for the decoder. It will filter out the mask points"""

        if self.config.mask_ratio == 0:
            return x, batch

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)

        # add position encoding to the mask tokens
        mask_tokens[..., :self.pos_encoder.get_out_dim()] = mask_tokens[..., :self.pos_encoder.get_out_dim()] + self.pos_encoder(batch["points_xyz_masked"])

        # visualize_point_batch(batch["points_xyz_masked"])

        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        batch["points_xyz"] = batch["points_xyz_all"]
        if "src_key_padding_mask_orig" in batch:
            batch["src_key_padding_mask"] = batch["src_key_padding_mask_orig"]
        return x, batch

    def random_mask(self, x: torch.Tensor, batch: dict):
        """
        Perform per-sample random masking by per-sample shuffling.
        Taken from: https://github.com/facebookresearch/mae/blob/main/models_mae.py
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        # TODO also mask batch correctly, i.e. mask the same points in the batch
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.config.mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]

        return ids_keep, ids_mask, ids_restore

    def patch_mask(self, x: torch.Tensor, batch: dict):
        """
        Mask a ceratin amount of points. The result should be patches which are masked.
        """

        N, L, C = x.shape

        ratio = self.config.mask_ratio
        points = batch["points_xyz"]

        # select k points per batch randomly
        if self.config.mode == "patch":
            centroids = points[:, torch.randperm(points.shape[1])[:self.config.num_patches]]
        elif self.config.mode == "patch_fp":
            start_time = time.time()
            centroids = self.furthest_point_sampling(points, self.config.num_patches)
            print(f"FPS took {time.time() - start_time}")
        else:
            raise ValueError(f"Unknown masking mode {self.config.mode}")
        # visualize_point_batch(centroids)

        # compute pairwise distances between points and centroids
        distances = torch.cdist(points, centroids)
        distances = torch.min(distances, dim=2)[0]
        distances, indices = torch.sort(distances, dim=1)

        len_keep = int(L * (1 - ratio))
        ids_restore = torch.argsort(indices, dim=1)
        indices_keep = indices[:, :len_keep]
        indices_mask = indices[:, len_keep:]

        return indices_keep, indices_mask, ids_restore

    def furthest_point_sampling(self, points, K):
        B, N, _ = points.shape

        centroids = torch.zeros(B, K, 3, device=points.device)
        distance = torch.ones(B, N, device=points.device) * 1e10
        farthest = torch.randint(0, N, (B,), device=points.device)

        batch_indices = torch.arange(B, device=points.device)
        for i in range(K):
            centroids[:, i] = points[batch_indices, farthest]
            centroid = centroids[:, i].unsqueeze(1)
            dist = torch.sum((points - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]

        return centroids

@dataclass
class FeatureGeneratorTorchConfig(InstantiateConfig):
    _target: type = field(default_factory=lambda: FeatureGeneratorTorch)

    use_rgb: bool = False
    """Should the rgb be used as a feature?"""
    out_rgb_dim: int = 16
    """The output dimension of the rgb feature"""

    use_density: bool = True
    """Should the density be used as a feature?"""
    out_density_dim: int = 2

    use_pos_encoding: bool = True
    """Should the position encoding be used as a feature?"""

    use_dir_encoding: bool = False
    """Should the direction encoding be used as a feature?"""

    use_normal_encoding: bool = False
    """Should the direction encoding be used as a feature?"""

    rot_augmentation: bool = True
    """Should the random rot augmentation around the z axis be used?"""

    jitter: float = 0.0
    """How much jitter should be used in the augmentation?"""

    jitter_clip: float = 10000.0
    """At what value the jitter should be clipped."""

    random_scale: float = 1.0
    """random scaling down of the point cloud. 1.0 means no scaling, 0.5 means all points are getting scaled by 0.5 down. Can't be bigger than 1.0"""

    visualize_point_batch: bool = False
    """Visualize the points of the batch? Useful for debugging"""

    log_point_batch: bool = True
    """Log the pointcloud to wandb? Useful for debugging. Happens in chance 1/5000"""

    pos_encoder: Literal["sin", "rff"] = "sin"
    """what kind of feature encoded should be used?"""


class FeatureGeneratorTorch(nn.Module):
    """Takes in a batch of b Ray bundles, samples s points along the ray. Then it outputs n x m x f matrix.
    Each row corresponds to one feature of a sampled point of the ray.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, config: FeatureGeneratorTorchConfig, aabb: TensorType[2, 3]):
        super().__init__()

        self.config: FeatureGeneratorTorchConfig = config

        self.aabb = Parameter(aabb, requires_grad=False)
        self.aabb = cast(TensorType[2, 3], self.aabb)

        if self.config.use_rgb:
            self.rgb_linear = nn.Sequential(
                nn.Linear(3, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self.config.out_rgb_dim),
            )

        if self.config.use_density:
            self.density_linear = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, self.config.out_density_dim),
            )

        if self.config.use_pos_encoding:
            if self.config.pos_encoder == "sin":
                self.pos_encoder = NeRFEncoding(
                    in_dim=3, num_frequencies=8, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
                )
            elif self.config.pos_encoder == "rff":
                self.pos_encoder = RFFEncoding(in_dim=3, num_frequencies=8, scale=10)
            else:
                raise ValueError(f"Unknown pos encoder {self.config.pos_encoder}")
        if self.config.use_dir_encoding or self.config.use_normal_encoding:
            self.dir_encoder = SHEncoding()

        self.learned_mask_value = torch.nn.Parameter(torch.randn(self.get_out_dim())*0.5)

    def forward(self, field_outputs: dict, transform_batch: dict):
        """
        Takes a ray bundle filters out non dense points and returns a feature matrix of shape [num_dense_samples, feature_dim]
        used_samples be 1 if surface sampling is enabled ow used_samples = num_ray_samples

        Input:
        - ray_bundle: RayBundle [N]

        Output:
         - out: [1, points, feature_dim]
         - weights: [N, num_ray_samples, 1]
         - density_mask: [N, used_samples]
         - misc:
            - rgb: [N, used_samples, 3]
            - density: [N, used_samples, 1]
            - ray_samples: [N, used_samples]
        """
        device = transform_batch["points_xyz"].device
        transform_batch["points_xyz_orig"] = transform_batch["points_xyz"].clone()

        encodings = []
        time1 = time.time()
        if self.config.use_rgb:
            rgb = field_outputs[FieldHeadNames.RGB]
            assert not torch.isnan(rgb).any()
            assert not torch.isinf(rgb).any()

            rgb_out = self.rgb_linear(rgb)

            assert not torch.isnan(rgb_out).any()
            assert not torch.isinf(rgb_out).any()

            encodings.append(rgb_out)

        time2 = time.time()

        if self.config.use_density:
            density = field_outputs[FieldHeadNames.DENSITY]
            # normalize density between 0 and 1
            density = (density - density.min()) / (density.max() - density.min())
            # assert no nan and no inf values
            # assert not torch.isnan(density).any()
            # assert not torch.isinf(density).any()
            if torch.isnan(density).any():
                CONSOLE.print("density has nan values: ", torch.isnan(density).sum())
                density[torch.isnan(density)] = 0.0
            if torch.isinf(density).any():
                CONSOLE.print("density has inf values: ", torch.isinf(density).sum())
                density[torch.isinf(density)] = 1000000.0

            assert not torch.isnan(density).any()
            density = self.density_linear(density)
            assert not torch.isnan(density).any()
            encodings.append(density)

        time3 = time.time()

        if self.config.rot_augmentation and self.training:
            batch_size = transform_batch["points_xyz"].shape[0]
            angles_np = np.random.uniform(0, 2 * np.pi, size=(batch_size,)).astype('f')
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
            ).to(transform_batch["points_xyz"].device)

        else:
            rot_matrix = torch.eye(3, device=device).unsqueeze(0)

        time4 = time.time()
        transform_batch["rot_mat"] = rot_matrix
        positions = transform_batch["points_xyz"]

        if self.config.rot_augmentation:
            # TODO consider turning that off if not self.training()
            positions = torch.matmul(positions, rot_matrix)
            # normalize postions to be within scene bounds  self.aabb: Tensor[2,3]
            # TODO if needed add the normalizeation for now hardcode clamp
            # positions = torch.clamp(positions, self.aabb[0], self.aabb[1])

        if self.config.jitter != 0.0 and self.training:
            jitter = torch.randn_like(positions) * self.config.jitter
            # clip the jitter
            jitter = torch.clamp(jitter, -self.config.jitter_clip, self.config.jitter_clip)
            positions = positions + jitter

        time5 = time.time()
        positions = self.normalize_positions(positions)
        time6 = time.time()

        positions = cast(TensorType, positions)
        # The positions need to be in [0,1] for the positional encoding
        positions_normalized = SceneBox.get_normalized_positions(positions, self.aabb)

        if self.config.random_scale != 1.0 and self.training:
            scale = torch.rand(1, device=device) * (1.0 - self.config.random_scale) + self.config.random_scale
            positions_normalized = positions_normalized * scale
            assert torch.all(positions_normalized >= 0.0)
            assert torch.all(positions_normalized <= 1.0)

        transform_batch["points_xyz"] = positions_normalized

        # assert that the points are between 0 and 1
        assert torch.all(positions_normalized >= 0.0)
        assert torch.all(positions_normalized <= 1.0)

        # normalize positions at 0 mean and within unit ball
        # mean = torch.mean(positions_normalized, dim=1).unsqueeze(1)
        # dist = torch.norm(positions_normalized - mean, dim=-1).max()
        time7 = time.time()

        assert ((not self.config.use_normal_encoding) or FieldHeadNames.NORMALS in field_outputs)
        if FieldHeadNames.NORMALS in field_outputs or FieldHeadNames.PRED_NORMALS in field_outputs:
            normals = field_outputs[FieldHeadNames.PRED_NORMALS] if FieldHeadNames.PRED_NORMALS in field_outputs else field_outputs[FieldHeadNames.NORMALS]
            if self.config.rot_augmentation:
                # TODO consider turning that off if not self.training()
                normals = torch.matmul(normals, rot_matrix)
            transform_batch["normals"] = normals
            if self.config.use_normal_encoding:
                normals = get_normalized_directions(normals)
                transform_batch["normals_encoded"] = normals
                encodings.append(self.dir_encoder(normals))
        time8 = time.time()
        if self.config.use_pos_encoding:
            pos_encoding = self.pos_encoder(positions_normalized)
            assert not torch.isnan(pos_encoding).any()
            encodings.append(pos_encoding)

        time9 = time.time()
        if self.config.use_dir_encoding:
            directions = transform_batch["directions"]
            directions = torch.nn.functional.normalize(directions, dim=-1)
            if self.config.rot_augmentation:
                # TODO consider turning that off if not self.training()
                directions = torch.matmul(directions, rot_matrix)
            directions = get_normalized_directions(directions)
            dir_encoding = self.dir_encoder(directions)

            assert not torch.isnan(dir_encoding).any()
            encodings.append(dir_encoding)

        time10 = time.time()
        out = torch.cat(encodings, dim=-1)
        # out: 1, num_dense, out_dim
        # weights: num_rays, num_samples, 1

        if "src_key_padding_mask" in transform_batch and transform_batch["src_key_padding_mask"] is not None:
            out[transform_batch["src_key_padding_mask"]] = self.learned_mask_value

        # positions_normalized = (positions_normalized - mean) / dist
        if self.config.visualize_point_batch:
            if "normals" in transform_batch:
                # visualize_point_batch(transform_batch["points_xyz"], normals=transform_batch["normals"])
                visualize_point_batch(transform_batch["points_xyz"])

            else:
                visualize_point_batch(transform_batch["points_xyz"])
            a = input("press enter to continue...")

        if self.config.log_point_batch and random.random() < (1 / 5000):
            log_points_to_wandb(transform_batch["points_xyz"])

        time11 = time.time()
        CONSOLE.print("FeatureGenerator - time1: ", time2 - time1)
        CONSOLE.print("FeatureGenerator - time2: ", time3 - time2)
        CONSOLE.print("FeatureGenerator - time3: ", time4 - time3)
        CONSOLE.print("FeatureGenerator - time4: ", time5 - time4)
        CONSOLE.print("FeatureGenerator - time5: ", time6 - time5)
        CONSOLE.print("FeatureGenerator - time6: ", time7 - time6)
        CONSOLE.print("FeatureGenerator - time7: ", time8 - time7)
        CONSOLE.print("FeatureGenerator - time8: ", time9 - time8)
        CONSOLE.print("FeatureGenerator - time9: ", time10 - time9)
        CONSOLE.print("FeatureGenerator - time10: ", time11 - time10)
        return out, transform_batch

    def normalize_positions(self, points: torch.tensor) -> torch.tensor:
        # normalize points to be within 0, 1 for x,y
        min_points = torch.min(points, dim=-2).values
        max_points = torch.max(points, dim=-2).values

        scene_range = self.aabb[1] - self.aabb[0]
        scene_min = self.aabb[0]

        x_range = (max_points[:, 0] - min_points[:, 0]).unsqueeze(-1)
        y_range = (max_points[:, 1] - min_points[:, 1]).unsqueeze(-1)
        z_range = (max_points[:, 2] - min_points[:, 2]).unsqueeze(-1)

        # choose the biggest range for normalization
        range = torch.max(torch.max(x_range, y_range), z_range)

        # points[:, :, 0] = (points[:, :, 0] - min_points[:, 0].unsqueeze(-1)) / range
        # points[:, :, 1] = (points[:, :, 1] - min_points[:, 1].unsqueeze(-1)) / range
        # points[:, :, 2] = (points[:, :, 2] - min_points[:, 2].unsqueeze(-1)) / range

        points = (points - min_points.unsqueeze(1)) / range.unsqueeze(-1)

        points = points * scene_range + scene_min

        # check that the points are within the scene bounds
        assert torch.all(points >= self.aabb[0])
        assert torch.all(points <= self.aabb[1])

        return points


    def get_out_dim(self) -> int:
        total_dim = 0
        total_dim += self.config.out_rgb_dim if self.config.use_rgb else 0
        total_dim += self.config.out_density_dim if self.config.use_density else 0
        total_dim += self.pos_encoder.get_out_dim() if self.config.use_pos_encoding else 0
        total_dim += self.dir_encoder.get_out_dim() if self.config.use_dir_encoding else 0
        total_dim += self.dir_encoder.get_out_dim() if self.config.use_normal_encoding else 0
        return total_dim



@dataclass
class TranformerEncoderModelConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: TransformerEncoderModel)

    num_layers: int = 6
    """The number of encoding layers in the feature transformer."""
    num_heads: int = 8
    """The number of multihead attention heads in the feature transformer."""
    dim_feed_forward: int = 64
    """The dimension of the feedforward network model in the feature transformer."""
    dropout_rate: float = 0.2
    """The dropout rate in the feature transformer."""
    feature_dim: int = 64
    """The number of layers the transformer scales up the input dimensionality to the sequence dimensionality."""


class TransformerEncoderModel(torch.nn.Module):
    def __init__(
        self,
        config: TranformerEncoderModelConfig,
        input_size: int,
        activation: Union[Callable, None] = None,
    ):
        super().__init__()
        self.config = config
        self.input_size = input_size
        self.activation = activation

        # Feature dim layer
        self.feature_dim_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, self.config.feature_dim),
            torch.nn.ReLU(),
        )

        # Define the transformer encoder
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            self.config.feature_dim,
            self.config.num_heads,
            self.config.dim_feed_forward,
            self.config.dropout_rate,
            batch_first=True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, self.config.num_layers)


    def forward(self, x, batch: dict):
        """
        Args: X: {batch_size, seq_len, input_size}
        """

        src_key_padding_mask = batch.get("src_key_padding_mask", None)

        x = self.feature_dim_layer(x)  #

        # Apply the transformer encoder. Last step is layer normalization
        x = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )  # {1, num_dense_samples, feature_dim}

        if self.activation is not None:
            x = self.activation(x)

        return x

    def get_out_dim(self) -> int:
        return self.config.feature_dim



@dataclass
class PointNetWrapperConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: PointNetWrapper)

    out_feature_channels: int = 128
    """The number of features the model should output"""

    radius_scale: float = 0.25
    """Pointnet has radiuses tuned for the unit sphere. This scales the radiuses to the scene size as our radius are at most sqrt(2) and in practice even smaller"""


class PointNetWrapper(nn.Module):
    def __init__(
        self,
        config: PointNetWrapperConfig,
        input_size: int,
        activation: Union[Callable, None] = None,
    ):
        """
        input_size: the true input feature size, i.e. the number of features per point. Internally the points will be prepended with the featuers.
        """
        super().__init__()
        self.config = config
        self.input_size = input_size
        self.activation = activation

        # PointNet takes xyz + features as input
        self.feature_transformer = get_model(
            num_classes=config.out_feature_channels, in_channels=input_size + 3, radius_factor=self.config.radius_scale
        )
        self.output_size = config.out_feature_channels

    def forward(self, x: torch.Tensor, batch: dict):
        start_time = time.time()
        # prepend points xyz to points features
        x = torch.cat((batch["points_xyz"], x), dim=-1)
        x = x.permute(0, 2, 1)
        x, l4_points = self.feature_transformer(x)
        if self.activation is not None:
            x = self.activation(x)
        CONSOLE.print("PointNetWrapper forward time: ", time.time() - start_time)

        return x

    def get_out_dim(self) -> int:
        return self.output_size


@dataclass
class StratifiedTransformerWrapperConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: StratifiedTransformerWrapper)

    # refer to the stratfied transformer for these parameters
    downsample_scale: int = 8
    depths: List[int] = field(default_factory=lambda: [2, 2, 6, 2])
    channels: List[int] = field(default_factory=lambda: [48, 96, 192, 384])
    num_heads: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    window_size: int = 4  # the multiple of the grid size used as window size
    up_k: int = 3  # is not used
    k: int = 16  # kernel size in the maxpooling downsample layer
    grid_size: float = 0.04 / 10  # how big the grid is
    quant_size: float = 0.01 / 10# TODO check what this does/means
    rel_query: bool = True  # use a lookuptable for the relative position
    rel_key: bool = True
    rel_value: bool = True
    drop_path_rate: float = 0.3
    concat_xyz: bool = True
    ratio: float = 0.25 # number of centroids in the downsampled point cloud
    sigma: float = 1  # influence distance of a single point (sigma * grid_size)
    num_layers: int = 4
    stem_transformer: bool = True  # what kind of model to use. one is downsampling ones isnt
    patch_size: int = 1  # TODO check what this does/means
    max_num_neighbors: int = 34
    # load_dir: str = "/data/vision/polina/projects/wmh/dhollidt/documents/Stratified-Transformer/weights/s3dis_model_best.pth"
    load_dir: str = ""
    """If using a pretrained model you can specify it here. The pipeline should load it. The pretrained weights will be overwritten by nesf weights if trained already."""


class StratifiedTransformerWrapper(nn.Module):
    def __init__(
        self,
        config: StratifiedTransformerWrapperConfig,
        input_size: int,
        activation: Union[Callable, None] = None,
    ):
        super().__init__()
        self.config = config
        self.activation = activation

        patch_size = self.config.grid_size * self.config.patch_size
        window_sizes = [patch_size * self.config.window_size * (2**i) for i in range(self.config.num_layers)]
        grid_sizes = [patch_size * (2**i) for i in range(self.config.num_layers)]
        quant_sizes = [self.config.quant_size * (2**i) for i in range(self.config.num_layers)]
        CONSOLE.print("patch_size", patch_size)
        CONSOLE.print("window_sizes", window_sizes)
        CONSOLE.print("grid_sizes", grid_sizes)
        CONSOLE.print("quant_sizes", quant_sizes)

        self.model = Stratified(
            self.config.downsample_scale,
            self.config.depths,
            self.config.channels,
            self.config.num_heads,
            window_sizes,
            self.config.up_k,
            grid_sizes,
            quant_sizes,
            rel_query=self.config.rel_query,
            rel_key=self.config.rel_key,
            rel_value=self.config.rel_value,
            drop_path_rate=self.config.drop_path_rate,
            concat_xyz=self.config.concat_xyz,
            num_classes=-1,
            ratio=self.config.ratio,
            k=self.config.k,
            prev_grid_size=self.config.grid_size,
            sigma=1.0,
            num_layers=self.config.num_layers,
            stem_transformer=self.config.stem_transformer,
            features_in_dim=input_size,
        )

        if self.config.load_dir != "":
            state_dict = torch.load(self.config.load_dir)["state_dict"]

            total_parameters_state_dict = sum(p.numel() for p in state_dict.values())

            # replace keys starting with module with model, _tables do not fit shapes
            state_dict = {key.replace("module", "model"): value for key, value in state_dict.items() }
            if input_size != 3:
                state_dict = {key.replace("module", "model"): value for key, value in state_dict.items() if not key.endswith("_table")}

            parameters_filtered = sum(p.numel() for p in state_dict.values())

            CONSOLE.print("State dict has ", parameters_filtered, " parameters out of ", total_parameters_state_dict, " parameters")

            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            CONSOLE.print("Loaded feature transformer from pretrained checkpoint")
            CONSOLE.print("Feature Transformer missing keys", missing_keys)
            CONSOLE.print("Feature Transformer unexpected keys", unexpected_keys)


    def forward(self, x: torch.Tensor, batch: dict):
        def batch_for_stratified_point_transformer(points, features):
            batch_size = points.shape[0]
            seq_len = points.shape[1]
            points = points.reshape(-1, 3)
            features = features.reshape(-1, features.shape[-1])
            offsets = torch.arange(seq_len, (1 + batch_size) * seq_len, seq_len, dtype=torch.int32)

            offset_ = offsets.clone()
            offset_[1:] = offset_[1:] - offset_[:-1]
            batch = torch.cat([torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0).long()
            batch = batch.to(x.device)

            sigma = 1.0
            radius = 2.5 * self.config.grid_size * sigma
            neighbor_idx = tp.ball_query(
                radius, self.config.max_num_neighbors, points, points, mode="partial_dense", batch_x=batch, batch_y=batch
            )[0]

            # get the neighbour_idx
            offsets = offsets.to(points.device)
            return points, features, offsets, batch, neighbor_idx

        feature_shape = x.shape
        points, features, offsets, batch_s, neighbour_idx = batch_for_stratified_point_transformer(
            points=batch["points_xyz"], features=x
        )
        # print("Coordinates range", torch.min(points, dim=0)[0].p, torch.max(points, dim=0)[0].p, torch.max(points, dim=0)[0].norm().item())
        # print("points shape", points, "features shape", features.shape, "x shape", x.shape)
        # print("dtypes", features.dtype, points.dtype, offsets.dtype, batch_s.dtype, neighbour_idx.dtype)
        x = self.model(features, points, offsets, batch_s, neighbour_idx)

        if self.activation is not None:
            x = self.activation(x)

        x = x.reshape(*feature_shape[:-1], -1)
        return x

    def get_out_dim(self) -> int:
        return self.config.channels[0]



@dataclass
class FieldTransformerConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: FieldTransformer)

    transformer: TranformerEncoderModelConfig = TranformerEncoderModelConfig(
        num_layers = 2,
        num_heads= 2,
    )

    mode: Literal["mean", "transformer"] = "mean"

    knn: int = 64

class FieldTransformer(nn.Module):
    def __init__(self, config: FieldTransformerConfig, input_size: int, activation: Union[Callable, None] = None):
        super().__init__()
        self.config = config
        self.activation = activation

        self.transformer = self.config.transformer.setup(input_size=input_size+3)
        self.head = nn.Linear(self.transformer.get_out_dim(), input_size)

        self.learnable_query_vector = nn.Parameter(torch.randn(input_size + 3))

    def forward(self, query_pos: torch.Tensor, neural_feat: torch.Tensor, neural_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_pos: [N', 3]
            neural_feat: [N, C]
            neural_pos: [N, 3]

        Returns:
            [N', C] tensor
        """

        assert neural_feat.shape[0] == neural_pos.shape[0], "neural_feat and neural_pos must have the same count of points"
        CONSOLE.print("Querying ", query_pos.shape[0], "points in neural field of ", neural_pos.shape[0])
        closest_ind, closest_dists = self.get_k_closest_points(query_pos, neural_pos)  # shape: [N', k]

        # Get the features of the k closest points
        closest_feat = neural_feat[closest_ind]  # shape: [N', k, C]
        closest_points = neural_pos[closest_ind]  # shape: [N', k, 3]

        if self.config.mode == "mean":
            inv_dist = 1.0 / (closest_dists + 1e-5)  # shape: [N', k]
            weighted_features = closest_feat * inv_dist.unsqueeze(-1)  # shape: [N', k, C]
            retrieved_feat = weighted_features.sum(dim=1)  # shape: [N', C]
        elif self.config.mode == "transformer":
            # compute the relative positions of the query points to the k closest points
            rel_pos =  closest_points - query_pos.unsqueeze(1) # shape: [N', k, 3]

            # concatenate the relative positions and the features of the k closest points
            rel_pos_feat = torch.cat([rel_pos, closest_feat], dim=-1)  # shape: [N', k, 3 + C]

            # prepend the learnable query vector
            rel_pos_feat = torch.cat([rel_pos_feat, self.learnable_query_vector.unsqueeze(0).expand(rel_pos_feat.shape[0], -1, -1)], dim=1)  # shape: [N', k + 1, 3 + C]
            # apply the transformer
            rel_pos_feat = self.transformer(rel_pos_feat, batch={})  # shape: [N', k + 1, C]
            # get the features of the query points
            retrieved_feat = rel_pos_feat[:, 0, :]  # shape: [N', C]

            retrieved_feat = self.head(retrieved_feat)  # shape: [N', OUT_DIM]

        return retrieved_feat


    def get_k_closest_points(self, query_pos, neural_pos):
        # Gets the k closests points for each quesry point
        # Compute Euclidean distance between each pair of points
        dists = torch.cdist(query_pos, neural_pos)  # shape: [N', N]

        # Find the indices of the k closest points
        dist, indices = torch.topk(dists, self.config.knn, largest=False, sorted=True, dim=-1)  # shape: [N', k]

        # dummy data
        # indices = torch.zeros(query_pos.shape[0], self.config.knn, dtype=torch.int64, device=query_pos.device)
        return indices, dist