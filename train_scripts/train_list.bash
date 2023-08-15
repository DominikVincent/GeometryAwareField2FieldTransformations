#!/bin/bash

set -x  # Enable debug mode to print each command

# Set the path to the directory containing the folders
# directory_path="/data/vision/polina/projects/wmh/dhollidt/datasets/klevr_nesf"
directory_path="/data/vision/polina/projects/wmh/dhollidt/datasets/toybox-5"

# Set ouput directory
# output_dir="/YOUR/PATH/HERE/klever_depth_normal_models_nesf"
output_dir="/YOUR/PATH/HERE/toybox-5-depth-normal-models-nesf-2"

# check or create output directory
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi

# Set the path to the bash script you want to execute
script_path="/YOUR/PATH/HERE/nerfstudio_fork/train_scripts/train.bash"

# folders_to_process=("39" "88" "113" "114" "116" "115" "131" "133" "134" "147" "155" "158" "161" "164" "160" "199" "203" "211" "217" "229" "230" "243" "246" "265" "266" "278")
folders_to_process=("281" "286" "284" "292" "293" "296" "299" "330" "385" "388" "394" "415" "418" "437" "432" "433" "441" "453" "461" "462" "470" "479" "483" "497" "502" "510" "514" "524")

for folder in "${folders_to_process[@]}"; do
  folder_path="$directory_path/$folder"

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
