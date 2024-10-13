from utils.dice import *
import pickle
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
from dipy.io.streamline import load_tractogram
import json
from dipy.io.stateful_tractogram import StatefulTractogram, Space
import sys
sys.path.append('/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/')
from utils.dice import hcp_scores
import ast
from tqdm import tqdm
from multiprocessing import Pool
import os


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
    sub, true_labels, predicted_labels, streamlines, int_label_mapping= args
    unique_labels = np.unique(true_labels)
    if len(unique_labels) == 0:
        return (sub, {})
    tractogram = load_tractogram(f"/scratch/anoushkrit.scee.iitmandi/DATA/dMRI_data/105HCP/test/{sub}/tracts/{int_label_mapping[unique_labels[0]]}.trk", "same", bbox_valid_check=False)
    dic = {}
    for label in unique_labels:
        true_streamlines = [s for s, t in zip(streamlines, true_labels) if t == label]
        pred_streamlines = [s for s, t in zip(streamlines, predicted_labels) if t == label]
        t1 = StatefulTractogram.from_sft(ArraySequence(true_streamlines), tractogram)
        t2 = StatefulTractogram.from_sft(ArraySequence(pred_streamlines), tractogram)
        dic[int_label_mapping[label]] = hcp_scores(t1, t2)
        
    return (sub, dic)

def calculate_hcp_scores(pickle_path, mapping_path, output_path):

    true_labels, predicted_labels, streamlines, subjects = load_pickle_data(pickle_path)
    print('----- pickle data loaded -----')
    subjects = np.array(subjects)
    unique_subjects = np.unique(subjects)
    
    with Pool() as p:
        streamlines = p.map(string_to_list, streamlines)
    print(len(streamlines))
    print(type(streamlines[0]), type(streamlines[0][0]))
    print('----- string to list done -----')

    int_label_mapping = get_label_mapping(mapping_path)

    inp = [] # creating input to pass to the function while multiprocessing
    for sub in unique_subjects:
        subject_streamlines = [s for s,t in zip(streamlines, subjects) if t == sub]
        true_labels_subject = [t for t, s in zip(true_labels, subjects) if s == sub]
        predicted_labels_subject = [t for t, s in zip(predicted_labels, subjects) if s == sub]
        inp.append((sub, true_labels_subject, predicted_labels_subject, subject_streamlines, int_label_mapping))


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
        print(output[unique_subjects[i]])
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)
    return output

if __name__ == "__main__":
    pickle_path = 'predictions/test_pred_single_streamline_105HCP.pkl'
    mapping_path = 'label_int_mapping.json'
    output_path = 'predictions/dice_scores_single_streamline_105HCP.json'
    calculate_hcp_scores(pickle_path, mapping_path, output_path)
