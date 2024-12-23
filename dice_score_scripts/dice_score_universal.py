import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dice import *    
import pickle
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
from dipy.io.streamline import load_tractogram
import json
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from utils.dice import hcp_scores
import ast
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd
import csv
from sklearn.metrics import classification_report
from tqdm import tqdm


def load_pickle_data(path):
    with open(path, 'rb') as f:
        pkl_data = pickle.load(f)

    return pkl_data

# def string_to_list(streamline):
#     list_str = ','.join(streamline.split())
#     clean_list_str = list_str.replace('[,', '[').replace(',]', ']')
#     clean_list_str = clean_list_str.replace(',,', ',')
#     streamline = np.array(ast.literal_eval(clean_list_str))
    
#     return streamline
def string_to_list(streamline):
    list_str = ','.join(streamline.split())
    clean_list_str = list_str.replace('[,', '[').replace(',]', ']')
    clean_list_str = clean_list_str.replace(', dtype=float32)', '').replace('array([', '[')
    clean_list_str = clean_list_str.replace(',,,,', ',').replace(',,,', ',').replace(',,', ',')
    clean_list_str = clean_list_str.replace(',', ', ')
    clean_list_str = clean_list_str.replace(', dtype=float32)', '').replace('array([', '[')
    streamline = list(ast.literal_eval(clean_list_str))
    
    return streamline

def get_label_mapping(path):
    with open(path, 'r') as f:
        label_int_mapping = json.load(f)
    int_label_mapping = {int(v): k for k, v in label_int_mapping.items()}
    return int_label_mapping
def calculate_hcp_scores_subject(args):
    sub, true_labels, predicted_labels, streamlines, int_label_mapping ,pickle_path= args
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
            continue
        dic[key]['precesion'] = report[key]['precision']
        dic[key]['recall'] = report[key]['recall']
        dic[key]['f1-score'] = report[key]['f1-score']
        dic[key]['support'] = report[key]['support']
    
    dic['accuracy'] = report['accuracy']
    dic['macro avg'] = report['macro avg']
    dic['weighted avg'] = report['weighted avg']

    return [sub, dic]

def calculate_hcp_scores(pickle_path, mapping_path, output_path, dataset_name):

    # laoding pickle data
    if 'streamline' in dataset_name:
        true_labels, predicted_labels, streamlines, subjects = load_pickle_data(pickle_path)
    elif 'qbx_goback_3L' in dataset_name:
        true_cluster_labels, predicted_cluster_labels, cluster_subjects, streamline_cluster_label, raw_streamlines_cluster = load_pickle_data(pickle_path)
    else:
        raise ValueError('Dataset name not recognized')
    print('----- pickle data loaded -----')

    # converting string to list
    if 'streamline' in dataset_name:
        subjects = np.array(subjects)
        unique_subjects = np.unique(subjects)
        with Pool() as p:
            streamlines = p.map(string_to_list, streamlines)
    elif 'qbx_goback_3L' in dataset_name:
        with Pool() as p:
            cluster_subjects = p.map(string_to_list, cluster_subjects)
        with Pool() as p:
            streamline_cluster_label = p.map(string_to_list, streamline_cluster_label)
        with Pool() as p:
            raw_streamlines_cluster = p.map(string_to_list, raw_streamlines_cluster)
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
        unique_subjects = np.unique(subjects)
    print('----- string to list done -----')

    # calculating accuracy
    eq = 0
    for i in range(len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
            eq += 1
    print(f'---- Streamline Accuracy {eq/len(true_labels)*100:.4f}% ----')

    # getting label mapping
    int_label_mapping = get_label_mapping(mapping_path)
    
    inp = [] # creating input to pass to the function while multiprocessing
    pickle_path = os.path.dirname(output_path)+'/true_pred_labels/'
    for sub in unique_subjects:
        subject_streamlines = [s for s,t in zip(streamlines, subjects) if t == sub]
        true_labels_subject = [t for t, s in zip(true_labels, subjects) if s == sub]
        predicted_labels_subject = [t for t, s in zip(predicted_labels, subjects) if s == sub]
        inp.append((sub, true_labels_subject, predicted_labels_subject, subject_streamlines, int_label_mapping, pickle_path))



    output = {}
    if os.path.isfile(output_path):
        with open(output_path, 'r') as f:
            output = json.load(f)
            
    for i in tqdm(range(len(inp))):
        output[unique_subjects[i]] = calculate_hcp_scores_subject(inp[i])
        output[unique_subjects[i]][0] = eq/len(true_labels)*100
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)
    return output

def convert_json_to_csv(json_dice, output_path):
    output_path = output_path.replace('.json', '')
    dataframes = {}

    for sub, dic in json_dice.items():
        dic = dic[1]
        for bundle, scores in dic.items():
            if bundle in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            for score_type, value in scores.items():
                if score_type not in dataframes.keys():
                    dataframes[score_type] = {}
                if sub not in dataframes[score_type].keys():
                    dataframes[score_type][sub] = {}
                dataframes[score_type][sub][bundle] = value
    
    for score_type, sub_dict in dataframes.items():
        df = pd.DataFrame(sub_dict)
        df.to_csv(f'{output_path}__{score_type}.csv')

# def dice_runner():
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a model with specified parameters.")
    
    # Add arguments
    parser.add_argument("--model", required=True, help="Name of the model to use.")
    parser.add_argument("--exp_name", required=True, help="Experiment name.")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset.")
    parser.add_argument("--mapping_path", required=True, help="Path to the label int mapping file.")
    
    # Parse arguments
    args = parser.parse_args()

    # print(args.model, args.exp_name, args.dataset_name, args.mapping_path)
    pickle_path = f'predictions_new/test_{args.exp_name}/{args.dataset_name}/pkls/test_pred__{args.model}.pkl'
    output_path = f'predictions_new/test_{args.exp_name}/{args.dataset_name}/results/test_pred__{args.model}.json'
    json_dice = calculate_hcp_scores(pickle_path, args.mapping_path, output_path, args.dataset_name)
    # with open (output_path, 'r') as f:
    #     json_dice = json.load(f)
    convert_json_to_csv(json_dice, output_path)
