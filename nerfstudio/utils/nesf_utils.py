from math import ceil, sqrt
from typing import Tuple, Union

import numpy as np
import plotly.graph_objs as go
import psutil
import torch
from rich.console import Console
from torchtyping import TensorType

import wandb
from nerfstudio.cameras.rays import RaySamples

CONSOLE = Console(width=120)

CLASS_TO_COLOR = (
    torch.tensor(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 255, 255],
            [128, 0, 0],
            [0, 128, 0],
            [0, 0, 128],
            [128, 128, 0],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
        ]
    )
    / 255.0
)


def sequential_batching(ray_samples: RaySamples, field_outputs: dict, batch_size: int):
    device = ray_samples.frustums.origins.device
    W = ray_samples.shape[0]

    padding = batch_size - (W % batch_size) if W % batch_size != 0 else 0
    for key, value in field_outputs.items():
        field_outputs[key] = torch.nn.functional.pad(value, (0, 0, 0, padding))

    masking_top = torch.full((W,), False, dtype=torch.bool)
    masking_bottom = torch.full((padding,), True, dtype=torch.bool)
    masking = torch.cat((masking_top, masking_bottom)).to(device)

    ids_shuffle = torch.arange(W + padding)
    ids_restore = ids_shuffle

    points = ray_samples.frustums.get_positions()
    directions = ray_samples.frustums.directions

    points_pad = torch.nn.functional.pad(points, (0, 0, 0, padding))
    directions_pad = torch.nn.functional.pad(directions, (0, 0, 0, padding))

    return field_outputs, masking, ids_shuffle, ids_restore, points_pad, directions_pad

def random_batching(ray_samples: RaySamples, field_outputs: dict, batch_size: int):
    device = ray_samples.frustums.origins.device
    W = ray_samples.shape[0]

    padding = batch_size - (W % batch_size) if W % batch_size != 0 else 0
    for key, value in field_outputs.items():
        field_outputs[key] = torch.nn.functional.pad(value, (0, 0, 0, padding))

    masking_top = torch.full((W,), False, dtype=torch.bool)
    masking_bottom = torch.full((padding,), True, dtype=torch.bool)
    masking = torch.cat((masking_top, masking_bottom)).to(device)

    ids_shuffle = torch.randperm(W + padding)
    ids_restore = torch.argsort(ids_shuffle)

    points = ray_samples.frustums.get_positions()
    directions = ray_samples.frustums.directions

    points_pad = torch.nn.functional.pad(points, (0, 0, 0, padding))
    directions_pad = torch.nn.functional.pad(directions, (0, 0, 0, padding))

    return field_outputs, masking, ids_shuffle, ids_restore, points_pad, directions_pad


def spatial_sliced_batching(ray_samples: RaySamples, field_outputs: dict, batch_size: int, scene_bound: TensorType[2,3]):
    """
    Given the features, density, points and batch size, order the spatially in such a away that each batch is a box on an irregualar grid.
    Each cell contains batch size points.
    In:
     - outs: [1, points, feature_dim]
     - density_mask: [N, used_samples]
     - misc["ray_samples]: [N, used_samples]

    Out:
     - outs: [1, points + padding_of_batch, feature_dim]
     - masking: [points + padding_of_batch]
     - ids_shuffle: [points + padding_of_batch]
     - ids_restore: [points + padding_of_batch]
     - points_pad: [points + padding_of_batch, 3]
     - colors_pad: [points + padding_of_batch, 3]
    """
    device = ray_samples.frustums.origins.device
    W = ray_samples.shape[0]

    columns_per_row = max(int(sqrt(W / batch_size)), 1)
    row_size = columns_per_row * batch_size
    row_count = ceil(W / row_size)

    CONSOLE.print(
        "Row count",
        row_count,
        "columns per row",
        columns_per_row,
        "Grid size",
        row_count * columns_per_row,
        "Takes points: ",
        row_count * columns_per_row * batch_size,
        "At point count: ",
        W,
    )

    padding = (
        (columns_per_row * batch_size) - (W % (columns_per_row * batch_size))
        if W % (columns_per_row * batch_size) != 0
        else 0
    )

    for key, value in field_outputs.items():
        field_outputs[key] = torch.nn.functional.pad(value, (0, 0, 0, padding))

    masking_top = torch.full((W,), False, dtype=torch.bool)
    masking_bottom = torch.full((padding,), True, dtype=torch.bool)
    masking = torch.cat((masking_top, masking_bottom)).to(device)

    points = ray_samples.frustums.get_positions()
    points_min = torch.min(points, dim=-2).values
    points_max = torch.max(points, dim=-2).values
    rand_points = torch.rand((padding, 3), device=points.device) * (points_max - points_min) + points_min
    # scale points to be closer to the scene bounds center by factor of 0.6
    # rand_points = (rand_points - scene_bound.mean(0)) * 0.6 + scene_bound.mean(0)
    
    points_padded = torch.cat((points, rand_points))

    directions = ray_samples.frustums.directions
    rand_directions = torch.randn((padding, 3), device=directions.device)
    directions_padded = torch.cat((directions, rand_directions))

    ids_shuffle = torch.argsort(points_padded[:, 0])

    for i in range(row_count):
        # for each row seperate into smaller cells by y axis
        row_idx = ids_shuffle[i * row_size : (i + 1) * row_size]
        points_row_y = points_padded[row_idx, 1]
        ids_shuffle[i * row_size : (i + 1) * row_size] = row_idx[torch.argsort(points_row_y)]

    ids_restore = torch.argsort(ids_shuffle)

    return field_outputs, masking, ids_shuffle, ids_restore, points_padded, directions_padded


def visualize_point_batch(points_pad: torch.Tensor, ids_shuffle: Union[None, torch.Tensor] = None, normals: torch.Tensor = None, classes: torch.Tensor = None):
    CONSOLE.print("Visualizing point batch")
    batch_size = points_pad.shape[1]

    if ids_shuffle is not None:
        points_pad = points_pad.view(-1, 3)[ids_shuffle, :].to("cpu")
        points_pad = points_pad.reshape(-1, batch_size, 3)
        if normals is not None:
            normals = normals.view(-1, 3)[ids_shuffle, :].to("cpu")
            normals = normals.reshape(-1, batch_size, 3)
    else:
        points_pad = points_pad.to("cpu")
        if normals is not None:
            normals = normals.to("cpu")
            
    # points pad reordered should be [B, batch_size, 3]
    if len(points_pad.shape) == 2:
        points_pad = points_pad.unsqueeze(0)
    
    if normals is not None and len(normals.shape) == 3:
        normals = normals.reshape(-1 ,3)

    if normals is not None:
        # normalize normals to length 0.01
        normals = (normals / torch.norm(normals, dim=-1, keepdim=True)) * 0.01
        
    points = torch.empty((0, 3))
    colors = torch.empty((0, 3))
    if classes is not None:
        classes = classes.to("cpu")
    for i in range(points_pad.shape[0]):
        points = torch.cat((points, points_pad[i, :, :]))
        
        if classes is not None:
            labels = classes[i, :]
            new_colors = torch.index_select(CLASS_TO_COLOR, 0, labels) * 255
        else:
            new_colors = torch.randn((1, 3)).repeat(batch_size, 1) * 255
            # new_colors = torch.zeros((1,3)).repeat(batch_size, 1)
        colors = torch.cat((colors, new_colors))
        
    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(color=colors, size=1, opacity=0.8),
    )
    data = [scatter]
    # create a layout with axes labels
    layout = go.Layout(
        title=f"RGB points: {points_pad.shape}",
        scene=dict(xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Z"), aspectmode="data"),
    )
    if normals is not None:
        # calculate end points for each line
        endpoints = points + normals
        n = points.shape[0]
        # create x and y arrays for each line
        x = np.c_[points[:,0], endpoints[:,0], np.full(n, np.nan)].flatten()
        y = np.c_[points[:,1], endpoints[:,1], np.full(n, np.nan)].flatten()
        z = np.c_[points[:,2], endpoints[:,2], np.full(n, np.nan)].flatten()
        normals = go.Scatter3d(x=x, y=y, z=z, mode='lines')
        data.append(normals)
    fig = go.Figure(data=data, layout=layout)
    # if normals is not None:
    #     fig.add_trace(
    #         go.Cone(
    #             x=points[:, 0],
    #             y=points[:, 1],
    #             z=points[:, 2],
    #             u=normals[:, 0],
    #             v=normals[:, 1],
    #             w=normals[:, 2],
    #             sizemode="absolute",
    #             sizeref=0.5,
    #             anchor="tail",
    #             showscale=False,
    #             opacity=0.5,
    #         )
    #     )
    fig.show()


def visualize_points(points: torch.Tensor):
    """ """
    points = points.squeeze(0).to("cpu").numpy()
    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(color="red", size=1, opacity=0.8),
    )

    # create a layout with axes labels
    layout = go.Layout(
        title=f"RGB points: {points.shape}",
        scene=dict(xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Z")),
    )
    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()


def log_points_to_wandb(points_pad: torch.Tensor, ids_shuffle: Union[None, torch.Tensor] = None):
    batch_size = points_pad.shape[1]
    
    if wandb.run is None:
        CONSOLE.print("[yellow] Wandb run is None but tried to log points. Skipping!")
        return
    if ids_shuffle is not None:
        points_pad_reordered = points_pad[ids_shuffle, :].to("cpu")
        points_pad_reordered = points_pad_reordered.reshape(-1, batch_size, 3)
    else:
        points_pad_reordered = points_pad.to("cpu")
    points = torch.empty((0, 3))
    colors = torch.empty((0, 3))
    for i in range(points_pad_reordered.shape[0]):
        points = torch.cat((points, points_pad_reordered[i, :, :]))
        colors = torch.cat((colors, torch.randn((1, 3)).repeat(batch_size, 1) * 255))
    point_cloud = torch.cat((points, colors), dim=1).detach().cpu().numpy()
    wandb.log({"point_samples": wandb.Object3D(point_cloud)}, step=wandb.run.step)


def compute_mIoU(confusion_matrix) -> Tuple[float, np.ndarray]:
    """columns is prediction rows is true, confusion needs to be unnormalized"""
    intersection = np.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(axis=0)
    predicted_set = confusion_matrix.sum(axis=1)
    union = ground_truth_set + predicted_set - intersection
    
    
    IoU = intersection / union.astype(np.float32)

    mIoU = np.nanmean(IoU)
    
    return mIoU, IoU

def calculate_accuracy(confusion_matrix):
    """columns is prediction rows is true, confusion matrix needs to be unnormalized"""
    
    # Get the diagonal elements of the confusion matrix
    correct_predictions = np.diag(confusion_matrix)
    # Get the total number of predictions
    total_predictions = np.sum(confusion_matrix)
    # Calculate the accuracy
    accuracy = np.sum(correct_predictions) / total_predictions
    accuracy_per_class = correct_predictions / np.sum(confusion_matrix, axis=1).astype(np.float32)
    return accuracy, accuracy_per_class

def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()

    # Return the memory usage in megabytes (MB)
    return mem_info.rss / (1024**2)