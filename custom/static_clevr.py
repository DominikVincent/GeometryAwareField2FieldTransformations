import logging
import shutil
import sys
import traceback
from time import sleep

sys.path.extend(["/usr/lib/x86_64-linux-gnu", "/data/vision/polina/users/clintonw/anaconda3/envs/cuda11/lib"])
import os

os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/data/vision/polina/users/clintonw/anaconda3/envs/cuda11/lib"

# set os environment variable to point to the dataset directory. Useful in docker
if "DS_DIR" not in os.environ:
    # os.environ["DS_DIR"] = "/data/vision/polina/scratch/clintonw/datasets/kubric-public/"
    os.environ["DS_DIR"] = "/kubric_data"

print(os.listdir("/kubric_data"))

import os.path as osp

import bpy
import kubric as kb
import numpy as np
import pybullet as pb
from kubric.renderer import Blender
from kubric.simulator import PyBullet

OUT_DIR = "/out_dir/klevr"
logging.basicConfig(level=logging.INFO)



def main():
    pb_client = pb.connect(pb.DIRECT)

    # --- Some configuration values
    # the region in which to place objects [(min), (max)]
    SPAWN_REGION = [(-6, -6, 0.3), (6, 6, 5)]
    CLEVR_OBJECTS = ("cube", "cylinder", "sphere", "cone", "torus")
    KUBASIC_OBJECTS = (
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
    )

    # --- CLI arguments
    parser = kb.ArgumentParser()
    # Configuration for the objects of the scene
    parser.add_argument("--objects_set", choices=["clevr", "kubasic"], default="kubasic")
    parser.add_argument("--min_num_objects", type=int, default=3, help="minimum number of objects")
    parser.add_argument("--max_num_objects", type=int, default=7, help="maximum number of objects")
    # Configuration for the floor and background
    parser.add_argument("--floor_friction", type=float, default=0.3)
    parser.add_argument("--floor_restitution", type=float, default=0.5)
    # parser.add_argument("--background", choices=["clevr", "colored"],
    #                                         default="clevr")
    parser.add_argument("--backgrounds_split", choices=["train", "test"], default="train")

    # Configuration for the camera
    parser.add_argument("-c", "--camera", choices=["spiral", "random"], default="spiral")
    parser.add_argument("--start_id", default=0, type=int)

    # Configuration for the source of the assets
    parser.add_argument("--kubasic_assets", type=str, default=osp.expandvars("$DS_DIR/assets/KuBasic/KuBasic.json"))
    parser.add_argument(
        "--hdri_assets", type=str, default=osp.expandvars("$DS_DIR/assets/HDRI_haven/HDRI_haven.json")
    )  # "gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
    parser.add_argument("--gso_assets", type=str, default=osp.expandvars("$DS_DIR/assets/GSO/GSO.json"))
    parser.add_argument("--save_state", action="store_true")
    parser.set_defaults(save_state=False, frame_end=256, frame_rate=12, resolution=256, job_dir=OUT_DIR)

    parser.add_argument("-n", "--num_trajectories", type=int, default=100)
    parser.add_argument("-o", "--overwrite", action="store_true")
    FLAGS = parser.parse_args()
    base_dir = FLAGS.job_dir
    kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
    hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)
    train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.1)
    if FLAGS.backgrounds_split == "train":
        logging.info("Choosing from %d training backgrounds...", len(train_backgrounds))
    else:
        logging.info("Choosing from %d held-out backgrounds...", len(test_backgrounds))
    print("Choose background")
    for i in range(FLAGS.start_id, FLAGS.num_trajectories + FLAGS.start_id):
        # try adding the object 3 times
        try_num = 0
        print(i)
        while try_num < 3:
            try_num += 1
            try:
                    # sleep for 5s
                # sleep(5)
                print(" ############### Trajectory: ", i)
                # --- Common setups & resources
                FLAGS.job_dir = f"{base_dir}/{i}"
                print("Job dir: ", FLAGS.job_dir)
                # check if folder is empty and if it is delete it
                if osp.exists(FLAGS.job_dir) and len(os.listdir(FLAGS.job_dir)) == 0:
                    print("Job dir exists and is empty: ", FLAGS.job_dir)
                    shutil.rmtree(FLAGS.job_dir)

                if osp.exists(FLAGS.job_dir):
                    print("Job dir exists: ", FLAGS.job_dir)
                    if FLAGS.overwrite:
                        shutil.rmtree(FLAGS.job_dir)
                    else:
                        break

                scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
                print("setup done")
                simulator = PyBullet(scene, scratch_dir)
                print("simulator done")
                renderer = Blender(scene, scratch_dir, samples_per_pixel=64)
                print("renderer done")
                resolution = scene.resolution
                print("resolution done", resolution)

                # --- Populate the scene
                # background HDRI
                if FLAGS.backgrounds_split == "train":
                    hdri_id = rng.choice(train_backgrounds)
                else:
                    hdri_id = rng.choice(test_backgrounds)
                background_hdri = hdri_source.create(asset_id=hdri_id)
                logging.info("Using background %s", hdri_id)
                scene.metadata["background"] = hdri_id
                renderer._set_ambient_light_hdri(background_hdri.filename)
                print("background done")

                # Dome
                dome = kubasic.create(
                    asset_id="dome",
                    name="dome",
                    friction=FLAGS.floor_friction,
                    restitution=FLAGS.floor_restitution,
                    static=True,
                    background=True,
                )
                assert isinstance(dome, kb.FileBasedObject)
                scene += dome
                dome_blender = dome.linked_objects[renderer]
                texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
                texture_node.image = bpy.data.images.load(background_hdri.filename)
                print("dome done")

                # Camera
                logging.info("Setting up the Camera...")
                focal_length = 30.0  # focal length (mm)
                sensor_width = 32
                scene.camera = kb.PerspectiveCamera(focal_length=focal_length, sensor_width=sensor_width)
                print("camera done")

                frame_list = []
                num_frames = (FLAGS.frame_end + 1) - (FLAGS.frame_start)
                if FLAGS.camera == "spiral":
                    rotations = 6
                    R = 15 + np.random.randn(num_frames)
                    phis = np.linspace(0, rotations * 2 * np.pi, num_frames)
                    thetas = np.linspace(np.pi * 0.5, np.pi * 0.08, num_frames)
                    positions = np.stack(
                        [R * np.sin(thetas) * np.cos(phis), R * np.sin(thetas) * np.sin(phis), R * np.cos(thetas) + 0.1], axis=1
                    )
                    for frame in range(FLAGS.frame_start, FLAGS.frame_end + 1):
                        ix = frame - (FLAGS.frame_start)
                        scene.camera.position = positions[ix]
                        scene.camera.look_at((np.random.randn() * 0.2, np.random.randn() * 0.2, 0.7 + np.random.randn() * 0.2))
                        scene.camera.keyframe_insert("position", frame)
                        scene.camera.keyframe_insert("quaternion", frame)
                        frame_data = {
                            "file_path": "rgba_{:05d}.png".format(ix),
                            "transform_matrix": scene.camera.matrix_world.tolist(),
                        }
                        frame_list.append(frame_data)

                elif FLAGS.camera == "random":  # Random position in half-sphere-shell
                    for frame in range(FLAGS.frame_start, FLAGS.frame_end + 1):
                        ix = frame - (FLAGS.frame_start)
                        scene.camera.position = kb.sample_point_in_half_sphere_shell(
                            inner_radius=8.0, outer_radius=10.0, offset=0.1
                        )
                        scene.camera.look_at((np.random.randn() * 0.2, np.random.randn() * 0.2, 0.7 + np.random.randn() * 0.2))
                        scene.camera.keyframe_insert("position", frame)
                        scene.camera.keyframe_insert("quaternion", frame)
                        frame_data = {
                            "file_path": "rgba_{:05d}.png".format(ix),
                            "transform_matrix": scene.camera.matrix_world.tolist(),
                        }
                        frame_list.append(frame_data)
                print("Frames done")

                # Add random objects
                num_objects = rng.randint(FLAGS.min_num_objects, FLAGS.max_num_objects + 1)
                logging.info("Randomly placing %d objects:", num_objects)
                for i in range(num_objects):
                    if FLAGS.objects_set == "clevr":
                        # get the a random index of the clever objects
                        shape_index = rng.randint(0, len(CLEVR_OBJECTS))
                        shape_name = CLEVR_OBJECTS[shape_index]
                        size_label, size = kb.randomness.sample_sizes("clevr", rng)
                        color_label, random_color = kb.randomness.sample_color("clevr", rng)
                    else:  # FLAGS.object_set == "kubasic":
                        shape_index = rng.randint(0, len(KUBASIC_OBJECTS))
                        shape_name = KUBASIC_OBJECTS[shape_index]
                        size_label, size = kb.randomness.sample_sizes("uniform", rng)
                        size *= 2
                        color_label, random_color = kb.randomness.sample_color("uniform_hue", rng)

                    material_name = rng.choice(["metal", "rubber"])
                    obj = kubasic.create(
                        asset_id=shape_name,
                        scale=size,
                        name=f"{size_label} {color_label} {material_name} {shape_name}",
                        segmentation_id=shape_index + 1,
                    )
                    print("Object done")
                    assert isinstance(obj, kb.FileBasedObject)

                    if material_name == "metal":
                        obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=1.0, roughness=0.2, ior=2.5)
                        obj.friction = 0.4
                        obj.restitution = 0.3
                        obj.mass *= 2.7 * size**3
                    else:  # material_name == "rubber"
                        obj.material = kb.PrincipledBSDFMaterial(
                            color=random_color, metallic=0.0, ior=1.25, roughness=0.7, specular=0.33
                        )
                        obj.friction = 0.8
                        obj.restitution = 0.7
                        obj.mass *= 1.1 * size**3

                    obj.metadata = {
                        "shape": shape_name.lower(),
                        "size": size,
                        "size_label": size_label,
                        "material": material_name.lower(),
                        "color": random_color.rgb,
                        "color_label": color_label,
                    }
                    scene.add(obj)
                    print("object added done")
                    kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION, rng=rng)
                    obj.velocity = (0, 0, 0)

                    print("object moved done")
                    logging.info("        Added %s at %s", obj.asset_id, obj.position)

                print("Added all objects done.")
                logging.info("Rendering the scene ...")
                # layers = ['rgba', 'depth', 'segmentation', 'normal', 'object_coordinates']
                layers = ["rgba", "segmentation", "depth", "normal"]
                data_stack = renderer.render(return_layers=layers)
                print("render done")

                # --- Postprocessing
                kb.compute_visibility(data_stack["segmentation"], scene.assets)
                visible_foreground_assets = [
                    asset for asset in scene.foreground_assets if np.max(asset.metadata["visibility"]) > 0
                ]
                visible_foreground_assets = sorted(  # sort assets by their visibility
                    visible_foreground_assets, key=lambda asset: np.sum(asset.metadata["visibility"]), reverse=True
                )

                data_stack["segmentation"] = kb.adjust_segmentation_idxs(
                    data_stack["segmentation"], scene.assets, visible_foreground_assets
                )
                scene.metadata["num_instances"] = len(visible_foreground_assets)

                # Save to image files
                kb.write_image_dict(data_stack, output_dir)
                kb.post_processing.compute_bboxes(data_stack["segmentation"], visible_foreground_assets)

                # nerfstudio format
                if True:
                    kb.write_json(
                        filename=output_dir / "transforms.json",
                        data={
                            "fl_x": focal_length * resolution[0] / sensor_width,
                            "fl_y": focal_length * resolution[1] / sensor_width,
                            "cx": resolution[0] / 2,
                            "cy": resolution[1] / 2,
                            "h": resolution[0],
                            "w": resolution[1],
                            "k1": 0.0,
                            "aabb_scale": 8,
                            "camera_angle_x": scene.camera.field_of_view,
                            "frames": frame_list,
                        },
                    )

                # nerf / instant ngp format
                else:
                    kb.write_json(
                        filename=output_dir / "transforms.json",
                        data={
                            "aabb_scale": 2,
                            "scale": 0.18,
                            "offset": [0.5, 0.5, 0.5],
                            "camera_angle_x": scene.camera.field_of_view,
                            "frames": frame_list,
                        },
                    )

                # --- Metadata
                logging.info("Collecting and storing metadata for each object.")
                kb.write_json(
                    filename=output_dir / "metadata.json",
                    data={
                        "flags": vars(FLAGS),
                        "metadata": kb.get_scene_metadata(scene),
                        "camera": kb.get_camera_info(scene.camera),
                        "instances": kb.get_instance_info(scene, visible_foreground_assets),
                    },
                )

        #         shutil.rmtree(scratch_dir)
        #         kb.done()
            except Exception as e:
                print("Failure due to error: ", e)
                traceback.print_exc()
                # sleep(1)
                continue
        if try_num >=3:
            print("################# Failed to render scene", i, " after ", try_num, " tries. Skipping.")
    try:
        pb.disconnect()
    except Exception:  # pylint: disable=broad-except
        pass  # cleanup is already done


if __name__ == "__main__":
    print("Starting main")
    main()
