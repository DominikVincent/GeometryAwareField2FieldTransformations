import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from pytransform3d.transform_manager import TransformManager

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.ray_samplers import UniformSampler
from nerfstudio.utils import plotly_utils as vis


def query_one_point(ray_bundle, model, point_idx, uniform=False, num_samples=10):
    qp_before = model.proposal_sampler.num_nerf_samples_per_ray
    model.proposal_sampler.num_nerf_samples_per_ray = 1

    ray_b = ray_bundle[point_idx]
    print(ray_b.origins.shape)
    print(ray_b)

    # expand one dimension to ray_b.origins to make [1, 3] instead of [3]
    ray_b.origins = ray_b.origins.unsqueeze(0)
    print(ray_b.origins.shape)
    # expand one dimension to ray_b.directions to make [1, 3] instead of [3]
    ray_b.directions = ray_b.directions.unsqueeze(0)
    print(ray_b.directions.shape)



    colors = model.forward(ray_b)
    if not uniform:
        ray_sample, weights_list, ray_samples_list = model.proposal_sampler(ray_b, density_fns=model.density_fns)
    else:
        unfiorm_ray_sampler = UniformSampler()
        ray_sample, weights_list, ray_samples_list = unfiorm_ray_sampler(ray_b, num_samples=10)
    print(ray_sample)

    field_outputs = model.field(ray_sample, compute_normals=model.config.predict_normals)

    model.proposal_sampler.num_nerf_samples_per_ray = qp_before
    return field_outputs


def get_grid_samples(grid_ranges, num_samples, pipeline):
    """returns ray samples which are on a grid

    @param range: range of the grid 3x2 array of min and max values for dimension x, y, z, e.g. [[-1, 1], [-1, 1], [-1, 1]].append
    @param num_samples: number of samples per dimension
    """

    # Get all the origins for points on 0th layer of yz plane
    x_points = torch.linspace(grid_ranges[0][0], grid_ranges[0][1], num_samples)
    z_points = torch.linspace(grid_ranges[2][0], grid_ranges[2][1], num_samples)


    # get all coordinates of the grid spanned by x and z points in 2D array [[x1, 0, z1], [x2, 0, z1], [x3, 0, z1], ...

    # TODO use meshgrid
    origins = []
    for x_idx in range(num_samples):
        for z_idx in range(num_samples):
            origins.append(torch.tensor([x_points[x_idx], grid_ranges[1][0], z_points[z_idx]]))

    origins = torch.stack(origins)

    # repeat direction vector for all origins
    direction_vector = torch.tensor([0, 1, 0])
    directions = direction_vector.repeat(origins.shape[0], 1)

    camera_index = torch.tensor([0])
    # repeat camera index for all origins
    camera_index = camera_index.repeat(origins.shape[0], 1)

    pixel_area = torch.tensor([5.3914e-07])
    # repeat pixel area for all origins
    pixel_area = pixel_area.repeat(origins.shape[0], 1)

    nears = torch.tensor(0)
    fars = torch.tensor(grid_ranges[1][1] - grid_ranges[1][0])
    print(f"near: {nears}, far: {fars}")
    # repeat near and far for all origins
    nears = nears.repeat(origins.shape[0], 1)
    fars = fars.repeat(origins.shape[0], 1)

    print(f"origins: {origins.shape}, nears: {nears.shape}")

    ray_bundle = RayBundle(
        origins=origins,
        directions=directions,
        camera_indices=camera_index,
        pixel_area=pixel_area,
        # nears=nears,
        # fars=fars,
    )

    # unfiorm_ray_sampler = UniformSampler()
    # ray_sample = unfiorm_ray_sampler(ray_bundle, num_samples=num_samples)

    bins = torch.linspace(0, grid_ranges[1][1] - grid_ranges[1][0], num_samples)[..., None]
    ray_sample = ray_bundle.get_ray_samples(bin_starts=bins[:-1, :], bin_ends=bins[1:, :])

    device = pipeline.device
    ray_bundle = ray_bundle.to(device)
    ray_sample = ray_sample.to(device)
    return ray_sample, ray_bundle


def get_forward_grid_points(model, ray_grid_sample):
    field_output = model.field(ray_grid_sample, compute_normals=model.config.predict_normals)

    point_colors = torch.flatten(field_output[FieldHeadNames.RGB], start_dim=0, end_dim=1)
    point_densities = torch.flatten(field_output[FieldHeadNames.DENSITY], start_dim=0, end_dim=1)
    print("Point densitites max: ", point_densities.max().item())

    # print cound of inf and nan values
    print(f"Got {torch.isinf(point_densities).sum().item()} inf values, Got: {torch.isnan(point_densities).sum().item()} nan values")

    # hist plot with plotly of point densities with bin range 0 to 5000
    #fig = px.histogram(point_densities.detach().cpu().numpy(), range_x=[0, 5000])
    #fig.show()

    # replace all inf and nan with 0
    point_densities[torch.isinf(point_densities)] = 0
    point_densities[torch.isnan(point_densities)] = 0


    # normalize the densities to [0, 1]
    # point_densities = (point_densities - point_densities.min()) / (point_densities.max() - point_densities.min())

    # samples_origins = torch.flatten(ray_grid_sample.frustums.origins, start_dim=0, end_dim=1).cpu()
    # y_axis = torch.flatten(ray_grid_sample.frustums.starts, start_dim=0, end_dim=1).cpu()
    # samples_origins[:, 1] = y_axis.squeeze() + samples_origins[:, 1]
    samples_origins = ray_grid_sample.frustums.flatten().get_positions().cpu()

    # filter out all samples_origins n x 3, pointcolors n x 3 and densitites n x 1 where density < threshold
    threshold = 0.2
    samples_origins = samples_origins[point_densities.squeeze() > threshold]
    point_colors = point_colors[point_densities.squeeze() > threshold]
    point_densities = point_densities[point_densities.squeeze() > threshold]

    point_densities = point_densities / 10
    point_densities[point_densities > 1] = 1
    print("Point densities zero: ", (point_densities < 0.1).sum().item())
    # plotly hist plot of point densities
    # fig = px.histogram(point_densities.detach().cpu().numpy(), range_x=[0, 1])
    # fig.show()
    return samples_origins, point_colors, point_densities



def visualize_grid_samples(samples_origins, point_colors, point_densities):
    marker_colors = [f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{d.item():f})' for c, d in zip(point_colors, point_densities)]

    # visualize 2D nx3 array of points with plottly scatterplot
    scatter3d = go.Scatter3d(x=samples_origins[:, 0], y=samples_origins[:, 1], z=samples_origins[:, 2],
                                    mode='markers',
                                    marker=dict(
                                        size=3,
                                        color=marker_colors,                # set color to an array/list of desired values
                                        )
                                    )
    fig = go.Figure(data=[scatter3d])
    return fig

def plot_cameras(cameras):
    tm = TransformManager()

    # visualize all 3d poses with plotly
    for i in range(cameras.camera_to_worlds.shape[0]):
        # continue loop with 90%
        if np.random.rand() > 0.99:
            continue
        transformation = cameras.camera_to_worlds[i]
        # stack [0, 0, 0, 1] to the transformation matrix
        transformation = torch.cat([transformation, torch.tensor([0, 0, 0, 1]).view(1, 4)], dim=0)
        # print translation of camera
        # visualize the transformation
        tm.add_transform( f"cam_{i}", "world", transformation)
        # print translation of camera
        # print(f"Camera {i} translation: ", tm.get_transform(f"cam_{i}", "world")[:3, 3])


    ax = tm.plot_frames_in("world", s=0.1)
    plt.show()


def get_camera_samples(camera, camera_idx, num_samples, near_plane = 0, far_plane = 4, num_samples_ray = 10):
    ray_bundle = camera.generate_rays(camera_indices=0)

    ind_x = np.random.choice(ray_bundle.shape[0], num_samples, replace=True)
    ind_y = np.random.choice(ray_bundle.shape[1], num_samples, replace=True)

    # regular samples
    num_samples_per_axis = int(np.sqrt(num_samples))
    ind_x = np.linspace(0, ray_bundle.shape[0] - 1, num_samples_per_axis).astype(int)
    ind_y = np.linspace(0, ray_bundle.shape[1] - 1, num_samples_per_axis).astype(int)
    ind_x, ind_y = np.meshgrid(ind_x, ind_y)
    ind_x = ind_x.flatten()
    ind_y = ind_y.flatten()

    # select ray bundle elements by random indicies
    ray_bundle = ray_bundle[ind_x, ind_y]
    # ray_bundle = ray_bundle[random_ind_x, random_ind_y, :]

    bins = torch.linspace(near_plane, far_plane, num_samples_ray + 1)[..., None]
    ray_samples = ray_bundle.get_ray_samples(bin_starts=bins[:-1, :], bin_ends=bins[1:, :])
    return ray_bundle, ray_samples

def visualize_camera_rays(ray_bundle, ray_samples, ray_length = 4):

    vis_rays = vis.get_ray_bundle_lines(ray_bundle, color="teal", length=ray_length, width=1)

    fig = go.Figure(data=[vis_rays] + vis.get_frustums_mesh_list(ray_samples.frustums))

    fig.show()
