import os

import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image
from plotly.subplots import make_subplots

# Path to the parent folder containing the subfolders
parent_folder = "/data/vision/polina/projects/wmh/dhollidt/kubric_datasets/klevr"

# List to store the file paths of the PNG files
file_paths = []

# Iterate through each subfolder
for i in range(100):
    folder_path = os.path.join(parent_folder, str(i))
    # file_path = os.path.join(folder_path, "rgba_00000.png")
    file_path = os.path.join(folder_path, "normal_00000.png")

    # Check if the file exists
    if os.path.exists(file_path):
        file_paths.append(file_path)

# Create a list of image traces for each file path
image_traces = []
for file_path in file_paths:
    with Image.open(file_path) as img:
        image_array = img.convert("RGB")
        image_trace = go.Image(z=image_array)
        image_trace.update(xaxis='x', yaxis='y')
        image_traces.append(image_trace)

# Calculate the number of rows and columns
num_images = len(image_traces)
num_columns = 15
num_rows = num_images // num_columns
if num_images % num_columns != 0:
    num_rows += 1

# Create the subplot grid
fig = make_subplots(rows=num_rows, cols=num_columns, horizontal_spacing=0.00, vertical_spacing=0.00)

# Add image traces to the subplot grid
for i, image_trace in enumerate(image_traces, start=1):
    row = (i - 1) // num_columns + 1
    col = (i - 1) % num_columns + 1
    fig.add_trace(image_trace, row=row, col=col)

# Update the subplot layout
fig.update_layout(
    showlegend=False,
    autosize=True,
    margin=dict(l=0, r=0, t=0, b=0),
    # width=1200,
    # height=num_rows * 120  # Adjust the height based on the number of rows
)

fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)

# Display the figure in the browser
fig.show(renderer="browser")
