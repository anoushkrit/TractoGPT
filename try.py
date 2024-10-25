#%%
import json
import pandas as pd
import os

json_path = 'predictions_new/goback_3L_streamline_GPTM_105HCP/dice_scores_fusion_105HCP.json'
output_path = os.path.dirname(json_path)
with open(json_path, 'r') as f:
    json_data = json.load(f)
mapping_path = 'label_int_mappings/label_int_mapping_105HCP.json'
# mapping_path = 'label_int_mappings/int_label_mapping_hybrid.json'
with open(mapping_path, 'r') as f:
    label_int_mapping = json.load(f)
#     int_label_mapping = json.load(f)
# label_int_mapping = {v: k for k, v in int_label_mapping.items()}


for i in json_data.keys():
    json_data[i] = json_data[i][1]


# Flatten the JSON structure into a list of dictionaries
flattened_data = []
for subject_id, classes in json_data.items():
    row = {'subject_id': subject_id}
    for class_name, scores in classes.items():
        if class_name not in label_int_mapping.keys():
            continue
        for metric, value in scores.items():
            if(metric != 'dice'):
                continue
            row[f'{class_name}'] = value
    flattened_data.append(row)

# Create the DataFrame
df = pd.DataFrame(flattened_data)
df.set_index('subject_id', inplace=True)
df['subject_avg'] = df.mean(axis=1)
df.loc['class_avg'] = df.mean()
print(output_path+'/dice_scores.csv')
df.to_csv(f'{output_path}/dice_scores.csv')


flattened_data = []
for subject_id, classes in json_data.items():
    row = {'subject_id': subject_id}
    for class_name, scores in classes.items():
        if class_name not in label_int_mapping.keys():
            continue
        for metric, value in scores.items():
            if(metric != 'overlap'):
                continue
            row[f'{class_name}'] = value
    flattened_data.append(row)

# Create the DataFrame
df = pd.DataFrame(flattened_data)
df.set_index('subject_id', inplace=True)
df['subject_avg'] = df.mean(axis=1)
df.loc['class_avg'] = df.mean()
df.to_csv(f'{output_path}/overlap_scores.csv')


flattened_data = []
for subject_id, classes in json_data.items():
    row = {'subject_id': subject_id}
    for class_name, scores in classes.items():
        if class_name not in label_int_mapping.keys():
            continue
        for metric, value in scores.items():
            if(metric != 'overreach'):
                continue
            row[f'{class_name}'] = value
    flattened_data.append(row)

# Create the DataFrame
df = pd.DataFrame(flattened_data)
df.set_index('subject_id', inplace=True)
df['subject_avg'] = df.mean(axis=1)
df.loc['class_avg'] = df.mean()
df.to_csv(f'{output_path}/overreach_scores.csv')


flattened_data = []
for subject_id, classes in json_data.items():
    row = {'subject_id': subject_id}
    for class_name, scores in classes.items():
        if class_name not in label_int_mapping.keys():
            continue
        for metric, value in scores.items():
            if(metric != 'precesion'):
                continue
            row[f'{class_name}'] = value
    flattened_data.append(row)

# Create the DataFrame
df = pd.DataFrame(flattened_data)
df.set_index('subject_id', inplace=True)
df['subject_avg'] = df.mean(axis=1)
df.loc['class_avg'] = df.mean()
df.to_csv(f'{output_path}/precesion_scores.csv')


flattened_data = []
for subject_id, classes in json_data.items():
    row = {'subject_id': subject_id}
    for class_name, scores in classes.items():
        if class_name not in label_int_mapping.keys():
            continue
        for metric, value in scores.items():
            if(metric != 'recall'):
                continue
            row[f'{class_name}'] = value
    flattened_data.append(row)

# Create the DataFrame
df = pd.DataFrame(flattened_data)
df.set_index('subject_id', inplace=True)
df['subject_avg'] = df.mean(axis=1)
df.loc['class_avg'] = df.mean()
df.to_csv(f'{output_path}/recall_scores.csv')


flattened_data = []
for subject_id, classes in json_data.items():
    row = {'subject_id': subject_id}
    for class_name, scores in classes.items():
        if class_name not in label_int_mapping.keys():
            continue
        for metric, value in scores.items():
            if(metric != 'f1-score'):
                continue
            row[f'{class_name}'] = value
    flattened_data.append(row)

# Create the DataFrame
df = pd.DataFrame(flattened_data)
df.set_index('subject_id', inplace=True)
df['subject_avg'] = df.mean(axis=1)
df.loc['class_avg'] = df.mean()
df.to_csv(f'{output_path}/f1-score_scores.csv')
