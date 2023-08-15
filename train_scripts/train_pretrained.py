#!/usr/bin/env python

import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import yaml

import wandb

# from nerfstudio.utils import writer

# from scripts.eval import ComputePSNR

parser = argparse.ArgumentParser()
parser.add_argument("--slurm", help="Use slurm", action="store_true", default=False)
parser.add_argument("--partition", help="Which slurm partition to use", type=str, default="QRTX5000")
parser.add_argument("--proj_name", help="The wandb proj name", type=str, default="dhollidt/toybox-5-nesf")
parser.add_argument("--data", help="Path to a data config if not the default should be used.", type=str)
parser.add_argument("--only_last_layer", help="Set if only last layer should be trained", action="store_true")
parser.add_argument("--rays", help="If a different amount of rays is supposed to be used", type=int, default=None)
parser.add_argument("--ground_removal", help="If ground removal should be activated", type=bool, default=True)

parser.add_argument("--scratch", help="Train from scratch", action="store_true", default=False)
parser.add_argument("runs", nargs="+", help="The names of the wandb runs to evaluate", type=str)


LOG_PATH = Path("/data/vision/polina/projects/wmh/dhollidt/tmp/logs")

def rewrite_config(config_path: Path, name: str, data_config_path: Union[str, None] = None, scratch: bool = False):
    """Rewrites the config of a run from a pretrain config to a semantic config which loads the pretrained weights."""
    # extend path to /config.yml if necessary
    global args

    if config_path.is_dir():
        config_path = config_path / "config.yml"
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)

    checkpoint_dir = config.get_checkpoint_dir()

    config.timestamp = "{timestamp}"
    config.set_timestamp()
    if scratch:
        config.load_dir = None
    else:
        config.load_dir = checkpoint_dir
    config.load_pretrained_model = True
    # config.wandb_project_name = "toybox-5-nesf"

    # remove the pretraining from the model config
    config.pipeline.model.pretrain = False
    config.pipeline.model.mode = "semantics"
    config.pipeline.model.proximity_loss = True
    config.pipeline.model.sampler.surface_sampling = True
    config.pipeline.model.sampler.samples_per_ray = 24



    if args.only_last_layer:
        config.pipeline.model.only_last_layer = True

    if args.rays is not None:
        config.pipeline.datamanager.train_num_rays_per_batch = args.rays
        config.pipeline.datamanager.eval_num_rays_per_batch = args.rays
        config.pipeline.model.eval_num_rays_per_chunk = args.rays

    if args.ground_removal:
        config.pipeline.model.sampler.ground_removal_mode = "ransac"
        config.pipeline.model.sampler.ground_points_count = 5000000000
        config.pipeline.model.sampler.ground_tolerance = 0.008

    # update the data path if provided
    if data_config_path is not None and data_config_path != "":
        config.pipeline.datamanager.dataparser.data_config = Path(data_config_path)


    # update the save path
    config.base_dir = config.get_base_dir()
    config.wandb_run_name = name

    out_path = config_path.parent / "auto_semantic_config.yml"
    out_path.write_text(yaml.dump(config), "utf8")
    time.sleep(2)


@dataclass
class EvalRun:
    path: Path
    name: str
    eval_set: Union[None, Path] = None


def get_sweep_runs(sweep_id, project_name):
    api = wandb.Api()
    sweep = api.sweep(project_name + "/" + sweep_id)
    return sweep.runs


def get_runs(run_names: List["str"], project_name: str):
    name_filter = {"$or": [{"display_name": run} for run in run_names]}
    api = wandb.Api()
    runs = api.runs(project_name, filters=name_filter)
    return runs


def wandb_run_to_path(run):
    return (
        Path(run.config["output_dir"])
        / run.config["experiment_name"].strip("/")
        / run.config["method_name"]
        / run.config["timestamp"]
    )


def ns_train(config_path, name, use_slurm=False, partition="QRTX5000"):
    command = f"ns-train nesf --load_config {config_path}"

    if use_slurm:
        # save command in bash script as file
        BASH_SCRIPT = f"""\
#!/bin/bash
source /data/vision/polina/projects/wmh/dhollidt/conda/bin/activate
conda activate nerfstudio2

{command}
"""
        script_path = Path("/data/vision/polina/projects/wmh/dhollidt/tmp") / name
        with open(script_path, "w") as f:
            f.write(BASH_SCRIPT)
            # give execute permission
            os.chmod(script_path, 0o755)

        date_string = time.strftime("%Y_%m_%d_%I_%M_%p")
        std_out_log_file = LOG_PATH / (f"'{name}'" + "_" + date_string + ".out")
        std_err_log_file = LOG_PATH / (f"'{name}'" + "_" + date_string + ".err")
        command = f"sbatch -p {partition} --exclude=fennel --mem=40G --gres=gpu:1 -t 96:60:00 -o {std_out_log_file} -e {std_err_log_file} '{script_path}'"

    print("Running command: ", command)
    # Execute the command and capture the output
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Stream the output in real-time
    while True:
        output = process.stdout.readline().decode()
        error = process.stderr.readline().decode()
        if output == "" and error == "" and process.poll() is not None:
            break
        if output:
            print(output.strip())
        if error:
            print(error.strip())

    # Wait for the command to finish and get the return code
    return_code = process.wait()
    return return_code


def runs_eval(
    runs: List[str],
    project_name: str,
    use_slurm: bool = True,
    eval_config: Union[None, Path] = None,
    partition: str = "QRTX5000",
    data_config_path: Union[str, None] = None,
    args = None,
):
    wandb_runs = get_runs(run_names=runs, project_name=project_name)
    for i, wandb_run in enumerate(wandb_runs):
        path = wandb_run_to_path(wandb_run)
        prefix = "scratch_" if args.scratch else "pretrained_"
        eval_run = EvalRun(path, prefix + wandb_run.name, eval_config)
        rewrite_config(eval_run.path, eval_run.name, data_config_path, args.scratch)
        ns_train(eval_run.path / "auto_semantic_config.yml", eval_run.name, use_slurm=use_slurm, partition=partition)


if __name__ == "__main__":
    args = parser.parse_args()

    runs_eval(args.runs, args.proj_name, use_slurm=args.slurm, partition=args.partition, data_config_path=args.data, args=args)
