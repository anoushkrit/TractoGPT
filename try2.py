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

    if 'TrInF' in methods:
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
                
