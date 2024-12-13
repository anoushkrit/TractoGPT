import sys
sys.path.append('/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/')
from utils.dice import *
import pickle
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
from dipy.io.streamline import load_tractogram
import json
from dipy.io.stateful_tractogram import StatefulTractogram, Space
sys.path.append('/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/')
from utils.dice import hcp_scores
import ast
from tqdm import tqdm
from multiprocessing import Pool
import os
import pandas as pd
from sklearn.metrics import classification_report   


def load_pickle_data(path):
    with open(path, 'rb') as f:
        true_labels, predicted_labels, streamlines, subjects = pickle.load(f)

    return true_labels, predicted_labels, streamlines, subjects

def string_to_list(streamline):
    list_str = ','.join(streamline.split())
    clean_list_str = list_str.replace('[,', '[').replace(',]', ']')
    streamline = np.array(ast.literal_eval(clean_list_str))
    
    return streamline

def get_label_mapping(path):
    with open(path, 'r') as f:
        label_int_mapping = json.load(f)
    int_label_mapping = {int(v): k for k, v in label_int_mapping.items()}
    return int_label_mapping
def calculate_hcp_scores_subject(args):
    
    sub, true_labels, predicted_labels, pkls_path, TrInF_to_105HCP_mapping = args
    HCP_to_TrInF_mapping = {v: k for k, v in TrInF_to_105HCP_mapping.items()}
    print(HCP_to_TrInF_mapping)
    unique_common_labels = np.unique([p for p in predicted_labels if p in HCP_to_TrInF_mapping.keys() and HCP_to_TrInF_mapping[p] in true_labels])


    if 'sub' in sub:
        tractogram = load_tractogram(f"/scratch/anoushkrit.scee.iitmandi/DATA/TractoInferno/derivatives/testset/{sub}/tractography/{sub}__{true_labels[0]}.trk", "same", bbox_valid_check=False)
    else:
        tractogram = load_tractogram(f"/scratch/anoushkrit.scee.iitmandi/DATA/dMRI_data/105HCP/test/{sub}/tracts/{true_labels[0]}.trk", "same", bbox_valid_check=False)
    dic = {}
    for label in unique_common_labels:
        with open(f'{pkls_path}/{label}/pred_streamlines.pkl', 'rb') as f:
            pred_streamlines = pickle.load(f)
        with open(f'{pkls_path}/{HCP_to_TrInF_mapping[label]}/true_streamlines.pkl', 'rb') as f:
            true_streamlines = pickle.load(f)
        
        t1 = StatefulTractogram.from_sft(ArraySequence(true_streamlines), tractogram)
        t2 = StatefulTractogram.from_sft(ArraySequence(pred_streamlines), tractogram)
        if 'sub' in sub:
            dic[label] = hcp_scores(t1, t2)
        else:
            dic[label] = hcp_scores(t1, t2)
        # dic[label] = hcp_scores(t1, t2)
    
    # true_labels = [int_label_mapping[int(t)] for t in true_labels]
    # predicted_labels = [int_label_mapping[int(t)] for t in predicted_labels]

    # pickle_path = pickle_path + f'{sub}.pkl'
    # os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    # with open(pickle_path, 'wb') as f:
    #     pickle.dump([true_labels, predicted_labels], f)
    
    # report = classification_report(true_labels, predicted_labels, output_dict=True)

    # for key in report.keys():
    #     if key not in dic.keys():
    #         continue
    #     dic[key]['precesion'] = report[key]['precision']
    #     dic[key]['recall'] = report[key]['recall']
    #     dic[key]['f1-score'] = report[key]['f1-score']
    #     dic[key]['support'] = report[key]['support']
    
    # dic['accuracy'] = report['accuracy']
    # dic['macro avg'] = report['macro avg']
    # dic['weighted avg'] = report['weighted avg']
    
    return [sub, dic]

def calculate_hcp_scores(pkls_path, mapping_path, output_path):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(mapping_path, 'r') as f:
        TrInF_to_105HCP_mapping = json.load(f)
    HCP_to_TrInF_mapping = {v: k for k, v in TrInF_to_105HCP_mapping.items()}
    
    sub = pkls_path.split('/')[-1]

    true_labels = []
    predicted_labels = []
    for classes in os.listdir(pkls_path):
        if 'true_streamlines.pkl' in os.listdir(os.path.join(pkls_path, classes)):
            true_labels.append(classes)
        if 'pred_streamlines.pkl' in os.listdir(os.path.join(pkls_path, classes)):
            predicted_labels.append(classes)
    

    print(f"True Labels: {true_labels}")
    print(f"Predicted Labels: {predicted_labels}")

    scores = calculate_hcp_scores_subject((sub, true_labels, predicted_labels, pkls_path, TrInF_to_105HCP_mapping))

    with open(output_path, 'w') as f:
        json.dump({scores[0]:scores[1]}, f, indent=4)
def convert_json_to_csv(json_data, out_path):
    output_path = out_path.split('.')[0] + '.csv'
    for i in json_data.keys():
        json_data[i] = json_data[i][1]

    # Flatten the JSON structure into a list of dictionaries
    flattened_data = []
    for subject_id, classes in json_data.items():
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
    df.to_csv(output_path)
    
if __name__ == "__main__":
    pickle_path = 'predictions_new/true_pred_streamlines/single_GPTM_105HCP71__TrInF32/sub-1006'
    mapping_path = 'label_int_mappings/TrInF_to_105HCP.json'
    output_path = 'predictions_new/single_GPTM_105HCP71__TrInF32/dice_scores_single_105HCP71__TrInF32.json'
    json_dice = calculate_hcp_scores(pickle_path, mapping_path, output_path)
    # convert_json_to_csv(json_dice, output_path)
