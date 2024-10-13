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
import plotly.graph_objects as go
from multiprocessing import Pool
import os


def plot_3d_streamlines(streamlines, label):
    fig = go.Figure()
    
    # Loop over each streamline
    for streamline in streamlines:
        x, y, z = zip(*streamline)  # Unzip streamline points
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name=label))
    
    # Update layout
    fig.update_layout(
        title=label,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        showlegend=True
    )
    
    # Show plot
    fig.write_html(f'plots/{label}.html')
def load_pickle_data(path):
    with open(path, 'rb') as f:
        true_labels, predicted_labels, subjects, streamline_label, raw_streamlines = pickle.load(f)

    return true_labels, predicted_labels, subjects, streamline_label, raw_streamlines

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
    sub, true_labels, predicted_labels, streamlines, int_label_mapping= args
    unique_labels = np.unique(true_labels)
    if len(unique_labels) == 0:
        return (sub, {})
    tractogram = load_tractogram(f"/scratch/anoushkrit.scee.iitmandi/DATA/TractoInferno/derivatives/testset/{sub}/tractography/{sub}__{int_label_mapping[unique_labels[0]]}.trk", "same", bbox_valid_check=False)
    dic = {}
    for label in unique_labels:
        true_streamlines = [s for s, t in zip(streamlines, true_labels) if t == label]
        pred_streamlines = [s for s, t in zip(streamlines, predicted_labels) if t == label]
        # plot_3d_streamlines(true_streamlines, f'{sub}_{int_label_mapping[label]}_true')
        # plot_3d_streamlines(pred_streamlines, f'{sub}_{int_label_mapping[label]}_pred')
        t1 = StatefulTractogram.from_sft(ArraySequence(true_streamlines), tractogram)
        t2 = StatefulTractogram.from_sft(ArraySequence(pred_streamlines), tractogram)
        dic[int_label_mapping[label]] = hcp_scores(t1, t2)
        
    return (sub, dic)

def calculate_hcp_scores(pickle_path, mapping_path, output_path):

    true_cluster_labels, predicted_cluster_labels, cluster_subjects, streamline_cluster_label, raw_streamlines_cluster = load_pickle_data(pickle_path)
    print('----- pickle data loaded -----')

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

    eq = 0
    for i in range(len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
            eq += 1
    print(f'---- Accuracy ---- {eq/len(true_labels)*100:.4f}%')

    print(len(streamlines))
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
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)
    
    return output

if __name__ == "__main__":
    pickle_path = 'predictions/test_pred_neomethod_GPTS.pkl'
    mapping_path = 'label_int_mapping_TrInF.json'
    output_path = 'predictions/dice_scores_neomethod_GPTS.json'
    calculate_hcp_scores(pickle_path, mapping_path, output_path)
