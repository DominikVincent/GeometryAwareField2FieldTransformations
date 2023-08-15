#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import tyro
from rich.console import Console

import wandb
from nerfstudio.configs.base_config import LocalWriterConfig, LoggingConfig
from nerfstudio.utils import writer
from nerfstudio.utils.eval_utils import eval_setup

CONSOLE = Console(width=120)


@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")
    save_images: bool = False
    use_wandb: bool = False
    name: str = ""
    wandb_name: str = "klevr-results"

    def main(self) -> None:
        """Main function."""
        writer.setup_event_writer(self.use_wandb, is_tensorboard_enabled=False, log_dir="logs", wandb_project_name=self.wandb_name)
        if self.name != "" and self.use_wandb and wandb.run is not None:
            wandb.run.name = self.name
            wandb.run.tags = ["test", "eval"]
        writer.setup_local_writer(
            LoggingConfig(local_writer=LocalWriterConfig(enable=False)), max_iter=100000, banner_messages=["HERRO"]
        )

        config, pipeline, checkpoint_path = eval_setup(self.load_config)
        writer.put_config("config", dataclasses.asdict(config), step=0)
        print(config)
        assert self.output_path.suffix == ".json"
        if self.save_images:
            metrics_3d_dict = pipeline.get_average_eval_image_metrics(save_path=self.output_path.parent, log_to_wandb=self.use_wandb, miou_3d=True)
            metrics_dict = pipeline.get_average_eval_image_metrics(
                save_path=self.output_path.parent, log_to_wandb=self.use_wandb, miou_3d=False
            )
        else:
            # metrics_3d_dict = pipeline.get_average_eval_image_metrics(log_to_wandb=self.use_wandb, miou_3d=True)
            metrics_dict = pipeline.get_average_eval_image_metrics(log_to_wandb=self.use_wandb, miou_3d=False)

        # log the final results
        writer.put_dict("test", metrics_dict, step=1000000)
        writer.put_dict("test_3d", metrics_3d_dict, step=1000000)
        writer.write_out_storage()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
            "results_3d": metrics_3d_dict,
        }
        print(benchmark_info)
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}")

        if wandb.run is not None:
            wandb.finish()


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputePSNR).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputePSNR)  # noqa
