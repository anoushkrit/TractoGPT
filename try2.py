#%%
import pandas as pd
import json

# Sample JSON structure
# {
#     "subject1": {
#         "classA": {"dice": 0.9, "overreach": 0.8, "overlap": 0.7},
#         "classB": {"dice": 0.85, "overreach": 0.75, "overlap": 0.65}
#     },
#     "subject2": {
#         "classA": {"dice": 0.88, "overreach": 0.77, "overlap": 0.66},
#         "classC": {"dice": 0.92, "overreach": 0.81, "overlap": 0.71}
#     }
# }

# Load your JSON (replace this with actual file reading if necessary)
with open('predictions/dice_scores_qbx_goback_3L_GPTM.json', 'r') as f:
    data = json.load(f)
for i in data.keys():
    data[i] = data[i][1]

# Flatten the JSON structure into a list of dictionaries
flattened_data = []
for subject_id, classes in data.items():
    row = {'subject_id': subject_id}
    for class_name, scores in classes.items():
        for metric, value in scores.items():
            if(metric != 'dice'):
                continue
            row[f'{class_name}'] = value
    flattened_data.append(row)

# Create the DataFrame
df = pd.DataFrame(flattened_data)

# Set 'subject_id' as the index (row labels)
df.set_index('subject_id', inplace=True)

# Display the DataFrame
df.to_csv('predictions/dice_scores_qbx_goback_3L_GPTM.csv')

#%%
import h5py
def extract_hdf5_content(file, group='/'):
    contents = {}
    for key in file[group].keys():
        item = file[group][key]
        if isinstance(item, h5py.Dataset):
            # Extract dataset
            contents[key] = item[()]
            print(f"Extracted dataset: {group}{key} with shape {item.shape}")
        elif isinstance(item, h5py.Group):
            # Recursively extract groups
            print(f"Entering group: {group}{key}")
            contents[key] = extract_hdf5_content(file, group + key + '/')
    return contents

file_path = '/scratch/anoushkrit.scee.iitmandi/DATA/FIESTA/7562790/data_tractoinferno_hcp_qbx.hdf5'
f = h5py.File(file_path, 'r')
# all_contents = extract_hdf5_content(f)
print(f.keys())

print(sorted(f['thres'].keys()))


#%%
# f['thres']['7500'].value


def print_hdf5_tree(file, group='/'):
    print(group)
    for key in file[group].keys():
        item = file[group][key]
        if isinstance(item, h5py.Group):
            print_hdf5_tree(file, group + key + '/')

print_hdf5_tree(f)
#%%
import numpy as np
import plotly.graph_objs as go
fig = go.Figure()

# Add streamlines to the figure
for i in range(data.shape[0]):
    streamline = data[i]  # Each streamline is a y x 3 array
    fig.add_trace(go.Scatter3d(
        x=streamline[:, 0],
        y=streamline[:, 1],
        z=streamline[:, 2],
        mode='lines',
        name=f'Streamline {i + 1}'
    ))

# Set layout
fig.update_layout(
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    ),
    title='3D Streamlines'
)

# Show the figure
fig.show()
#%%
data.shape
#%%
import os

def rename_files_and_folders_recursively(folder_path):
    # Traverse all directories and files starting from folder_path
    for root, dirs, files in os.walk(folder_path, topdown=False):
        # Rename all files in the current directory
        for filename in files:
            if 'neomethod' in filename:
                new_filename = filename.replace('neomethod', 'qbx_goback_3L')
                old_file = os.path.join(root, filename)
                new_file = os.path.join(root, new_filename)
                os.rename(old_file, new_file)
                print(f"Renamed file: {old_file} -> {new_file}")

        # Rename all directories in the current directory
        for dirname in dirs:
            if 'neomethod' in dirname:
                new_dirname = dirname.replace('neomethod', 'qbx_goback_3L')
                old_dir = os.path.join(root, dirname)
                new_dir = os.path.join(root, new_dirname)
                os.rename(old_dir, new_dir)
                print(f"Renamed folder: {old_dir} -> {new_dir}")

# Specify the path to your folder
folder_path = '/scratch/anoushkrit.scee.iitmandi/github/TractoGPT'

# Call the function to rename files and folders recursively
rename_files_and_folders_recursively(folder_path)
