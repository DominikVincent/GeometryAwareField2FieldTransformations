#!/bin/bash



DATASET_PATH=/data/vision/polina/projects/wmh/dhollidt/kubric_datasets
DATASET_NAME=klevr
echo $DATASET_PATH

for ((h=1; h<=100; h++))
do
    # Execute command 1
    sudo docker run --rm --interactive --gpus '"device=1"' \
                --env KUBRIC_USE_GPU=1 \
                --volume "$(pwd):/kubric" \
                --volume "/data/vision/polina/scratch/clintonw/datasets/kubric-public:/kubric_data" \
                --volume "$DATASET_PATH:/out_dir" \
                --user $(id -u):$(id -g) \
                --volume "$PWD:/kubric" \
                kubricdockerhub/kubruntu_gpu \
                python3 static_clevr.py

    # Execute command 2
    for i in {0..99}; do [ -d "$DATASET_PATH/$DATASET_NAME/$i" ] && [ -z "$(ls -A "$DATASET_PATH/$DATASET_NAME/$i")" ] && rmdir "$DATASET_PATH/$DATASET_NAME/$i"; done

done