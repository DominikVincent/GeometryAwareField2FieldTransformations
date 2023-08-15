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
parser.add_argument(
    "--fix_config", help="whether to use wandb config to fix the local model config", action="store_true", default=False
)
parser.add_argument("--partition", help="Which slurm partition to use", type=str, default="QRTX5000")
parser.add_argument(
    "--eval_config",
    help="The nesf eval dataset file",
    type=str,
    default="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_normal_nesf_test_20.json",
)
parser.add_argument("--proj_name", help="The wandb proj name", type=str, default="dhollidt/toybox-5-nesf")

subparsers = parser.add_subparsers(dest="command")

sweep_parser = subparsers.add_parser("sweep")
sweep_parser.add_argument("--sweep_id", help="The wandb sweep id", type=str, default="kfsdevg7")

run_parser = subparsers.add_parser("run")
run_parser.add_argument("runs", nargs="+", help="The names of the wandb runs to evaluate", type=str)


FIX_CONFIG = False

LOG_PATH = Path("/data/vision/polina/projects/wmh/dhollidt/tmp/logs")


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


def sweep_run_to_path(run):
    return (
        Path(run.config["output_dir"])
        / run.config["experiment_name"].strip("/")
        / run.config["method_name"]
        / run.config["timestamp"]
    )


def ns_eval(config_path, output_path, name, use_slurm=False, partition="QRTX5000"):
    global args
    command = f"ns-eval --load-config {config_path} --output-path {output_path} --use-wandb --name '{name}' --wandb_name {args.proj_name}"

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
        command = f"sbatch -p {partition} --exclude=sumac --gres=gpu:1 -t 360:00 --mem-per-cpu 4000 -o {std_out_log_file} -e {std_err_log_file} '{script_path}'"

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


def dispatch_eval_run(run: EvalRun, use_slurm=False, wandb_config=None, partition="QRTX5000"):

    out_path = run.path / "auto_eval_config.yml"
    input_path = run.path / "config.yml"

    config = yaml.load(input_path.read_text(), Loader=yaml.Loader)

    if run.eval_set is not None:
        config.pipeline.datamanager.dataparser.data_config = run.eval_set

    if FIX_CONFIG:
        assert wandb_config is not None
        for key, value in wandb_config["pipeline"]["model"].items():
            if key.startswith("_"):
                continue
            if hasattr(config.pipeline.model, key):
                should_type = type(getattr(config.pipeline.model, key))
                cast_value = should_type(value)
                # print("Should be type: ", should_type, "cast to: ", type(cast_value))
                # print(f"Set {key} to {value} in model config from: {getattr(config.pipeline.model, key)}")
                setattr(config.pipeline.model, key, cast_value)
            else:
                print(f"WARNING: {key} not found in model config")
    # save config as yaml
    out_path.write_text(yaml.dump(config), "utf8")

    ns_eval(out_path, run.path / "auto_eval.json", name=run.name, use_slurm=use_slurm, partition=partition)
    print(" ######################### Done with: ", run, " ######################### ")


def sweep_eval(
    project_name: str,
    sweep_id: str,
    eval_config: Union[None, Path] = None,
    use_slurm: bool = True,
    partition: str = "QRTX5000",
):
    runs = get_sweep_runs(sweep_id, project_name)
    for i, run in enumerate(runs):
        path = sweep_run_to_path(run)
        eval_run = EvalRun(path, run.name + "_test", eval_config)
        dispatch_eval_run(eval_run, wandb_config=run.config, use_slurm=use_slurm, partition=partition)


def runs_eval(
    runs: List[str],
    project_name: str,
    use_slurm: bool = True,
    eval_config: Union[None, Path] = None,
    partition: str = "QRTX5000",
):
    wandb_runs = get_runs(run_names=runs, project_name=project_name)
    for i, wandb_run in enumerate(wandb_runs):
        path = sweep_run_to_path(wandb_run)
        eval_run = EvalRun(path, wandb_run.name + "_test", eval_config)

        dispatch_eval_run(eval_run, use_slurm=use_slurm, partition=partition)


if __name__ == "__main__":
    args = parser.parse_args()
    FIX_CONFIG = args.fix_config
    if args.command == "sweep":
        proj_name = args.proj_name
        sweep_id = args.sweep_id
        eval_config = Path(args.eval_config) if args.eval_config is not None or args.eval_config != "" else None
        sweep_eval(proj_name, sweep_id, eval_config, partition=args.partition, use_slurm=args.slurm)
    elif args.command == "run":
        eval_config = Path(args.eval_config) if args.eval_config is not None or args.eval_config != "" else None
        runs_eval(args.runs, args.proj_name, use_slurm=args.slurm, partition=args.partition, eval_config=eval_config)
