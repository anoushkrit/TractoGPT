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
with open('predictions/dice_scores_neomethod_GPTS.json', 'r') as f:
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
df.to_csv('predictions/dice_scores_neomethod_GPTS.csv')

#%%
import h5py
train_dict = {}
# Open the HDF5 file
with h5py.File('/scratch/anoushkrit.scee.iitmandi/DATA/FIESTA/7562790/data_tractoinferno_hcp_qbx.hdf5', 'r') as hdf:
    # Loop through all the datasets in 'train'
    for key in hdf['train'].keys():
        train_dict[key] = hdf['train'][key][:]  # Convert each dataset to a NumPy array

# Now 'train_dict' contains all datasets from 'train' group as a dictionary
print("Keys in the train_dict:", list(train_dict.keys()))