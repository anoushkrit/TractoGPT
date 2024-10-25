#%%
import pandas as pd
import os
import json

root_path = 'predictions_new'

accuracy = {}
for model_folder in os.listdir(root_path):
    model_path = os.path.join(root_path, model_folder)
    if not os.path.isdir(model_path):
        continue
    for json_file in os.listdir(model_path):
        if not json_file.endswith('.json'):
            continue
        with open(os.path.join(model_path, json_file)) as f:
            data = json.load(f)
        
        for sub in data.keys():
            if sub not in accuracy:
                accuracy[sub] = {}
            accuracy[sub][model_folder] = data[sub][1]['accuracy']*100

flattened_data = []
for subject_id, classes in accuracy.items():
    row = {'subject_id': subject_id}
    for model_name, scores in classes.items():
            row[f'{model_name}'] = scores
    flattened_data.append(row)

# Create the DataFrame
df = pd.DataFrame(flattened_data)
df.set_index('subject_id', inplace=True)
df.loc['method_avg'] = df.mean()
df.to_csv(f'{root_path}/accuracy_scores.csv')