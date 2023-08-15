import copy
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import wandb
from nerfstudio.configs.method_configs import method_configs

PROJECT = "mae-models-project"


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def debug():
    print("############## STARTING NEW SWEEP RUN #################")
    wandb.init(project=PROJECT)
    config = wandb.config
    print("############## INIT WANDB AND GOT CONFIG #################")

    wandb.log({"Eval Loss": 0.5}, step=0)
    wandb.log({"Eval Loss": 0.4}, step=1)
    wandb.log({"Eval Loss": 0.3}, step=2)
    wandb.log({"Eval Loss": 0.2}, step=3)
    wandb.log({"Eval Loss": 0.1}, step=4)
    time.sleep(3)
    wandb.finish()


def main():
    # reset_wandb_env()
    print("############## STARTING NEW SWEEP RUN #################")
    wandb.init(project=PROJECT)
    config = wandb.config
    print("############## INIT WANDB AND GOT CONFIG #################")

    data_config_path = Path("/YOUR/PATH/HERE/data/nesf_test_config_5.json")

    OUTPUT_DIR = Path("/YOUR/PATH/HERE/nesf_models/")
    DATA_PATH = Path("/data/vision/polina/projects/wmh/dhollidt/datasets/klevr/11")

    trainConfig = copy.deepcopy(method_configs["nesf"])
    trainConfig.vis = "wandb"
    trainConfig.data = DATA_PATH
    trainConfig.output_dir = OUTPUT_DIR
    trainConfig.pipeline.datamanager.dataparser.data_config = data_config_path
    trainConfig.steps_per_eval_all_images = 25000

    trainConfig.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    trainConfig.pipeline.datamanager.dataparser.data = trainConfig.data

    # update config with dict
    for key, value in config.items():
        if key.startswith("model"):
            key = key.replace("model_", "")
            if hasattr(trainConfig.pipeline.model, key):
                setattr(trainConfig.pipeline.model, key, value)
                print(f"Set {key} to {value} in model config")
            else:
                print(f"WARNING: {key} not found in model config")

    trainConfig.optimizers["feature_network"]["optimizer"].lr = config["lr"]
    trainConfig.optimizers["feature_transformer"]["optimizer"].lr = config["lr"]
    trainConfig.optimizers["learned_low_density_params"]["optimizer"].lr = config["lr"]
    trainer = trainConfig.setup(local_rank=0, world_size=1)

    trainConfig.save_config()
    trainer.setup()
    trainer.train()
    print("############## TRAINING COMPLETED #################")

    wandb.run.finish()
    print("############## WANDB RUN FINISHED #################")


# rgb
# sweep_configuration = {
#     "method": "random",
#     "metric": {"goal": "minimize", "name": "Eval Loss"},
#     "parameters": {
#         "lr": {"values": [1e-3]},
#         "model_rgb": {"value": True},
#         "model_rgb_feature_dim": {"values": [4, 8, 16]},
#         "model_use_feature_pos": {"values": [True, False]},
#         "model_use_feature_dir": {"values": [True, False]},
#         "model_feature_transformer_num_layers": {"values": [2, 4, 8]},
#         "model_feature_transformer_num_heads": {"values": [2, 4, 8]},
#         "model_feature_transformer_dim_feed_forward": {"values": [16, 32, 64, 128]},
#         "model_feature_transformer_feature_dim": {"values": [4, 8, 16, 32, 64, 128]},
#     },
# }

# semantic
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "Eval Loss"},
    "parameters": {
        "lr": {"value": 1e-3},
        "model_rgb": {"value": True},
        "model_pretrain": {"value": True},
        "model_mask_ratio": {"values": [0.2, 0.4, 0.6, 0.8]},
        "model_density_threshold": {"max": 2.0, "min": 0.0001},
        "model_rgb_feature_dim": {"values": [8, 16]},
        "model_feature_transformer_num_layers": {"values": [4, 6, 8]},
        "model_feature_transformer_num_heads": {"values": [4, 8]},
        "model_feature_transformer_dim_feed_forward": {"values": [64, 128]},
        "model_feature_transformer_feature_dim": {"values": [32, 64, 128, 256]},
    },
}

if __name__ == "__main__":
    print(datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    args = sys.argv
    if len(args) > 1:
        sweep_id = args[1]
        wandb.agent(sweep_id, function=main, count=5)
    else:
        sweep_id = wandb.sweep(sweep_configuration, project=PROJECT)
        print("sweep_id:", sweep_id)
