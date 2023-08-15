#!/bin/bash

# PARTITION=A6000
PARTITION=QRTX5000
# PARTITION=gpu
# PARTITION=titan

sbatch -p $PARTITION --gres=gpu:1 -t 120:60:00 --mem=30G -o logs/$(date +"%Y_%m_%d_%H_%M_%p").out -e logs/$(date +"%Y_%m_%d_%H_%M_%p").err $1
