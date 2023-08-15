#!/bin/bash

set -x  # Enable debug mode to print each command

# Set the path to the directory containing the folders
# directory_path="/data/vision/polina/projects/wmh/dhollidt/datasets/klevr_nesf"
# directory_path="/data/vision/polina/projects/wmh/dhollidt/datasets/toybox-5"
directory_path="/data/vision/polina/projects/wmh/dhollidt/kubric_datasets/klevr"

# Set ouput directory
# output_dir="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/klever_depth_normal_models_nesf"
output_dir="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/klevr-normal"

# check or create output directory
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi

# Set the path to the bash script you want to execute
script_path="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/nerfstudio_fork/train_scripts/train.bash"

# Set the number of folders to process
num_folders=1
start_folder=12

folder_list=($(ls -1 "$directory_path" | grep -E "^[0-9]+$" | sort -n))

# Iterate through each folder in the directory
count=0
dir_counter=0
for folder in "${folder_list[@]}"; do
  folder_path="$directory_path/$folder"
  if [ $count -eq $num_folders ]; then
    break
  fi
  ((dir_counter++))


  # Skip folders before the start folder
  if [ $dir_counter -le $start_folder ]; then
    continue
  fi

  # get the folder name
  echo "$folder_path"

  folder_name=$(basename "$folder_path")




  if [ -d "$folder_path" ] && [ "$(ls -A "$folder_path")" ]; then
    # Execute the script with the folder name as an argument
    $script_path "$folder_path" "$folder_name" "$output_dir"
    # echo "$folder_path"
    ((count++))
  fi
done
