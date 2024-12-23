import sys
sys.path.append('./TractoGPT/')
from utils.dice import *
import pickle
import argparse
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
from dipy.io.streamline import load_tractogram
import json
from dipy.io.stateful_tractogram import StatefulTractogram, Space
sys.path.append('./TractoGPT/')
from utils.dice import hcp_scores
import ast
from tqdm import tqdm
from multiprocessing import Pool
import os
from sklearn.metrics import classification_report


def load_pickle_data(path):
    with open(path, 'rb') as f:
        pkl_data = pickle.load(f)

    return pkl_data

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
    sub, true_labels, predicted_labels, streamlines, int_label_mapping, pickle_path = args
    unique_labels = np.unique(true_labels)
    if len(unique_labels) == 0:
        return (sub, {})
    if 'sub' in sub:
        tractogram = load_tractogram(f"/scratch/anoushkrit.scee.iitmandi/DATA/TractoInferno/derivatives/testset/{sub}/tractography/{sub}__{int_label_mapping[unique_labels[0]]}.trk", "same", bbox_valid_check=False)
    else:
        tractogram = load_tractogram(f"/scratch/anoushkrit.scee.iitmandi/DATA/dMRI_data/105HCP/test/{sub}/tracts/{int_label_mapping[unique_labels[0]]}.trk", "same", bbox_valid_check=False)
    dic = {}
    for label in unique_labels:
        true_streamlines = [s for s, t in zip(streamlines, true_labels) if t == label]
        pred_streamlines = [s for s, t in zip(streamlines, predicted_labels) if t == label]
        t1 = StatefulTractogram.from_sft(ArraySequence(true_streamlines), tractogram)
        t2 = StatefulTractogram.from_sft(ArraySequence(pred_streamlines), tractogram)
        dic[int_label_mapping[label]] = hcp_scores(t1, t2)
        
    true_labels = [int_label_mapping[t] for t in true_labels]
    predicted_labels = [int_label_mapping[t] for t in predicted_labels]

    pickle_path = pickle_path + f'{sub}.pkl'
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    with open(pickle_path, 'wb') as f:
        pickle.dump([true_labels, predicted_labels], f)

    report = classification_report(true_labels, predicted_labels, output_dict=True)

    for key in report.keys():
        if key not in dic.keys():
            print(key)
            continue
        dic[key]['precesion'] = report[key]['precision']
        dic[key]['recall'] = report[key]['recall']
        dic[key]['f1-score'] = report[key]['f1-score']
        dic[key]['support'] = report[key]['support']
    
    dic['accuracy'] = report['accuracy']
    dic['macro avg'] = report['macro avg']
    dic['weighted avg'] = report['weighted avg']

    return [sub, dic]

def calculate_hcp_scores(pickle_path, mapping_path, output_path):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    true_labels, predicted_labels, streamlines, subjects = load_pickle_data(pickle_path)
    print('----- pickle data loaded -----')
    subjects = np.array(subjects)
    unique_subjects = np.unique(subjects)
    
    with Pool() as p:
        streamlines = p.map(string_to_list, streamlines)
        
    print('----- string to list done -----')

    int_label_mapping = get_label_mapping(mapping_path)

    inp = [] # creating input to pass to the function while multiprocessing
    pickle_path = os.path.dirname(output_path)+'/true_pred_labels/'
    for sub in unique_subjects:
        subject_streamlines = [s for s,t in zip(streamlines, subjects) if t == sub]
        true_labels_subject = [t for t, s in zip(true_labels, subjects) if s == sub]
        predicted_labels_subject = [t for t, s in zip(predicted_labels, subjects) if s == sub]
        inp.append((sub, true_labels_subject, predicted_labels_subject, subject_streamlines, int_label_mapping, pickle_path))


    # with Pool() as p:
    #     results = p.map(calculate_hcp_scores_subject, inp)

    # output = {}
    # for sub, dic in results:
    #     output[sub] = dic
    output = {}
    if os.path.isfile(output_path):
        with open(output_path, 'r') as f:
            output = json.load(f)
    for i in range(len(inp)):
        output[unique_subjects[i]] = calculate_hcp_scores_subject(inp[i])
        # print(output[unique_subjects[i]])
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a model with specified parameters.")
    
    # Add arguments
    parser.add_argument("--model", required=True, help="Name of the model to use.")
    parser.add_argument("--exp_name", required=True, help="Experiment name.")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset.")
    parser.add_argument("--mapping_path", required=True, help="Path to the label int mapping file.")
    
    # Parse arguments
    args = parser.parse_args()

    pickle_path = f'predictions_new/{args.exp_name}/pkls/test_pred__{args.dataset_name}__{args.model}.pkl'
    mapping_path = args.mapping_path
    output_path = f'predictions_new/{args.exp_name}/single_GPTM_105HCP__TrInF/dice_scores_single_streamline_105HCP__TrInF.json'
    calculate_hcp_scores(pickle_path, mapping_path, output_path)
