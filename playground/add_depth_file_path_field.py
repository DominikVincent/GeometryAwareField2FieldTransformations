import json
import os
import shutil
from pathlib import Path

BASE_PATH=Path("/data/vision/polina/projects/wmh/dhollidt/kubric_datasets/klevr")
# Iterate through the folders
for i in range(100):
    directory = BASE_PATH / str(i)
    file_path = os.path.join(directory, "transforms.json")
    # backup_path = os.path.join(directory, "transforms_bk.json")

    # If the file exists
    if os.path.exists(file_path):
        # Rename the original file as backup
        # shutil.copy(file_path, backup_path)

        # Open and load the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Add depth_file_path field for each frame
        for frame in data['frames']:
            num = frame['file_path'].split('_')[1].split('.')[0]  # Extract the number
            frame['depth_file_path'] = f"depth_{num}.tiff"  # Construct the depth file path
            frame["normal_file_path"] = f"normal_{num}.png"  # Construct the normal file path

        # Write the updated data back to the JSON file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        print(f"File not found: {file_path}")
