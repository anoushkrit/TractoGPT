#%%
import pickle
import os
import json
import ast
import numpy as np
from multiprocessing import Pool
import gc

root_path = 'predictions_new/pkls'
save_path = 'predictions_new/true_pred_streamlines'
def string_to_list(streamline):
    list_str = ','.join(streamline.split())
    clean_list_str = list_str.replace('[,', '[').replace(',]', ']')
    clean_list_str = clean_list_str.replace(', dtype=float32)', '').replace('array([', '[')
    clean_list_str = clean_list_str.replace(',,,,', ',').replace(',,,', ',').replace(',,', ',')
    clean_list_str = clean_list_str.replace(',', ', ')
    clean_list_str = clean_list_str.replace(', dtype=float32)', '').replace('array([', '[')
    streamline = list(ast.literal_eval(clean_list_str))
    
    return streamline


for methods in os.listdir(root_path):
    print(methods)
    method_path = os.path.join(root_path, methods)
    if not method_path.endswith('.pkl'):
        continue
    
    if '105HCP__TrInF' in methods:
        with open('label_int_mappings/label_int_mapping_TrInF_23.json', 'r') as f:
            label_int_mapping = json.load(f)
        int_label_mapping = {v: k for k, v in label_int_mapping.items()}
    elif 'TrInF__105HCP' in methods:
        with open('label_int_mappings/label_int_mapping_105HCP_23.json', 'r') as f:
            label_int_mapping = json.load(f)
        int_label_mapping = {v: k for k, v in label_int_mapping.items()}
    elif 'TrInF' in methods:
        with open('label_int_mappings/label_int_mapping_TrInF.json', 'r') as f:
            label_int_mapping = json.load(f)
        int_label_mapping = {v: k for k, v in label_int_mapping.items()}
    elif '105HCP' in methods:
        with open('label_int_mappings/label_int_mapping_105HCP.json', 'r') as f:
            label_int_mapping = json.load(f)
        int_label_mapping = {v: k for k, v in label_int_mapping.items()}
    elif 'hybrid' in methods:
        with open('label_int_mappings/label_int_mapping_TrInF_25.json', 'r') as f:
            label_int_mapping = json.load(f)
        int_label_mapping = {v: k for k, v in label_int_mapping.items()}
    else:
        raise NotImplementedError(f'No label_int_mapping for {methods}')
    
    if 'qbx' not in methods:
        with open(method_path, 'rb') as f:
            true_labels, predicted_labels, streamlines, subjects = pickle.load(f)
        
        with Pool() as p:
            streamlines = p.map(string_to_list, streamlines)
        
        subjects = np.array(subjects)
        unique_subjects = np.unique(subjects)

        true_labels = [int_label_mapping[t] for t in true_labels]
        predicted_labels = [int_label_mapping[t] for t in predicted_labels]

        for sub in unique_subjects:
            if os.path.exists(f'{save_path}/{"_".join(methods.split(".")[0].split("_")[2:])}/{sub}'):
                continue
            if sub not in ['599469', 'sub-1006']:
                continue
            subject_streamlines = [s for s,t in zip(streamlines, subjects) if t == sub]
            true_labels_subject = [t for t, s in zip(true_labels, subjects) if s == sub]
            predicted_labels_subject = [t for t, s in zip(predicted_labels, subjects) if s == sub]

            unique_true_labels_subject = np.unique(true_labels_subject)
            for label in unique_true_labels_subject:
                actual_streamlines = [s for s,t in zip(subject_streamlines, true_labels_subject) if t == label]
                pred_streamlines = [s for s,t in zip(subject_streamlines, predicted_labels_subject) if t == label]

                pickle_path = f'{save_path}/{"_".join(methods.split(".")[0].split("_")[2:])}/{sub}/{label}'
                os.makedirs(pickle_path, exist_ok=True)
                with open(f'{pickle_path}/true_streamlines.pkl', 'wb') as f:
                    pickle.dump(actual_streamlines, f)
                with open(f'{pickle_path}/pred_streamlines.pkl', 'wb') as f:
                    pickle.dump(pred_streamlines, f)
        
        del true_labels, predicted_labels, streamlines, subjects
        gc.collect()
    elif 'qbx' in methods:
        with open(method_path, 'rb') as f:
            true_labels_str, predicted_cluster_labels, subjects_str, streamline_cluster_label_str, raw_streamlines_cluster_str = pickle.load(f)
    
        with Pool() as p:
            cluster_subjects = p.map(string_to_list, subjects_str)
        with Pool() as p:
            streamline_cluster_label = p.map(string_to_list, streamline_cluster_label_str)
        with Pool() as p:
            raw_streamlines_cluster = p.map(string_to_list, raw_streamlines_cluster_str)

        streamlines = []
        predicted_labels = []
        true_labels = []
        subjects = []
        for i in range(len(cluster_subjects)):
            for j in range(len(cluster_subjects[i])):
                streamlines.append(raw_streamlines_cluster[i][j])
                predicted_labels.append(predicted_cluster_labels[i])
                true_labels.append(streamline_cluster_label[i][j])
                subjects.append(cluster_subjects[i][j])
        subjects = np.array(subjects)

        true_labels = [int_label_mapping[t] for t in true_labels]
        predicted_labels = [int_label_mapping[t] for t in predicted_labels]

        unique_subjects = np.unique(subjects)
        for sub in unique_subjects:
            if os.path.exists(f'{save_path}/{"_".join(methods.split(".")[0].split("_")[2:])}/{sub}'):
                continue
            if sub not in ['599469', 'sub-1006']:
                continue
            subject_streamlines = [s for s,t in zip(streamlines, subjects) if t == sub]
            true_labels_subject = [t for t, s in zip(true_labels, subjects) if s == sub]
            predicted_labels_subject = [t for t, s in zip(predicted_labels, subjects) if s == sub]

            unique_true_labels_subject = np.unique(true_labels_subject)
            for label in unique_true_labels_subject:
                actual_streamlines = [s for s,t in zip(subject_streamlines, true_labels_subject) if t == label]
                pred_streamlines = [s for s,t in zip(subject_streamlines, predicted_labels_subject) if t == label]

                pickle_path = f'{save_path}/{"_".join(methods.split(".")[0].split("_")[2:])}/{sub}/{label}'
                os.makedirs(pickle_path, exist_ok=True)
                with open(f'{pickle_path}/true_streamlines.pkl', 'wb') as f:
                    pickle.dump(actual_streamlines, f)
                with open(f'{pickle_path}/pred_streamlines.pkl', 'wb') as f:
                    pickle.dump(pred_streamlines, f)
        
        del true_labels_str, predicted_cluster_labels, subjects_str, streamline_cluster_label_str, raw_streamlines_cluster_str
        del cluster_subjects, streamline_cluster_label, raw_streamlines_cluster
        del streamlines, predicted_labels, true_labels, subjects
        gc.collect()
    else:
        raise NotImplementedError(f'No label_int_mapping for {methods}')
                

#%%
# import json
# import os

# buans = ['predictions_new/buan1.json', 'predictions_new/buan2.json', 'predictions_new/buan3.json', 'predictions_new/buan4.json', 'predictions_new/buan5.json', 'predictions_new/buan6.json', 'predictions_new/buan7.json', 'predictions_new/buan8.json', 'predictions_new/buan9.json']

# save_path = 'predictions_new/buan.json'

# results = {}

# for buan in buans:
#     with open(buan, 'r') as f:
#         buan_results = json.load(f)
#     results.update(buan_results)

# results = dict(sorted(results.items()))

# for key in results.keys():
#     results[key] = dict(sorted(results[key].items()))

#     for sub in results[key].keys():
#         results[key][sub] = dict(sorted(results[key][sub].items()))

# with open(save_path, 'w') as f:
#     json.dump(results, f, indent=4)

#%%
import pandas
import json

buan_json = 'predictions_new/buan.json'

with open(buan_json, 'r') as f:
    buan_results = json.load(f)

buan_hcp = []
buan_tr = []

methods = list(buan_results.keys())

for i,method in enumerate(methods):
    for j,sub in enumerate(buan_results[method].keys()):
        temp_dict = {}
        temp_dict['method'] = method
        temp_dict.update(buan_results[method][sub])
        if sub == '599469':
            buan_hcp.append(temp_dict)
        elif sub == 'sub-1006':
            buan_tr.append(temp_dict)
        else:
            print('will not use this :)')

buan_hcp_df = pandas.DataFrame(buan_hcp)
buan_tr_df = pandas.DataFrame(buan_tr)

buan_hcp_df.to_csv('predictions_new/buan_hcp.csv', index=False)
buan_tr_df.to_csv('predictions_new/buan_tr.csv', index=False)

#%%
# import pandas as pd
# import json

# json_path = 'predictions_new/FIESTA/sub-1006.json'
# with open(json_path, 'r') as f:
#     fiesta_results = json.load(f)

# fiesta_scores = []
# dice = {}
# overlap = {}
# overreach = {}
# for class_ in fiesta_results.keys():
#     dice[class_] = fiesta_results[class_]['dice']
#     overlap[class_] = fiesta_results[class_]['overlap']
#     overreach[class_] = fiesta_results[class_]['overreach']

# fiesta_scores.append(dice)
# fiesta_scores.append(overlap)
# fiesta_scores.append(overreach)
# fiesta_df = pd.DataFrame(fiesta_scores, index=['Dice', 'Overlap', 'Overreach'])

# # Sort columns alphabetically
# fiesta_df = fiesta_df.reindex(sorted(fiesta_df.columns), axis=1)

# fiesta_df.to_csv('predictions_new/FIESTA/fiesta_scores.csv')
#%%
import pandas as pd
import json

json_path = 'predictions_new/single_GPTS_TrInF/dice_scores_single_streamline_TrInF.json'
with open(json_path, 'r') as f:
    fiesta_results = json.load(f)
fiesta_results = fiesta_results['sub-1006'][1]

fiesta_scores = []
dice = {}
overlap = {}
overreach = {}
precesion = {}
recall = {}
f1 = {}
support = {}
for class_ in fiesta_results.keys():
    if type(fiesta_results[class_]) != dict or 'dice' not in fiesta_results[class_].keys():
        continue
    dice[class_] = fiesta_results[class_]['dice']
    overlap[class_] = fiesta_results[class_]['overlap']
    overreach[class_] = fiesta_results[class_]['overreach']
    precesion[class_] = fiesta_results[class_]['precesion']
    recall[class_] = fiesta_results[class_]['recall']
    f1[class_] = fiesta_results[class_]['f1-score']
    support[class_] = fiesta_results[class_]['support']

fiesta_scores.append(dice)
fiesta_scores.append(overlap)
fiesta_scores.append(overreach)
fiesta_scores.append(precesion)
fiesta_scores.append(recall)
fiesta_scores.append(f1)
fiesta_scores.append(support)

fiesta_df = pd.DataFrame(fiesta_scores, index=['Dice', 'Overlap', 'Overreach', 'Precesion', 'Recall', 'F1-score', 'Support'])

# Sort columns alphabetically
fiesta_df = fiesta_df.reindex(sorted(fiesta_df.columns), axis=1)

fiesta_df.to_csv(os.path.join(os.path.dirname(json_path), 'scores_sub-1006.csv'))