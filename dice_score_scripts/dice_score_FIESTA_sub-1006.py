import sys
sys.path.append('./TractoGPT/')
from utils.dice import *
import pickle
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
import pandas as pd
import csv
from sklearn.metrics import classification_report
import plotly.graph_objects as go


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
    os.makedirs('Fiesta_plots_sub-1006', exist_ok=True)
    fig.write_html(f'Fiesta_plots_sub-1006/{label}.html')

def calculate_hcp_scores_subject(args):
    
    sub, gt, predicted, method = args
    predicted_trks = os.listdir(predicted)
    if method in ['FIESTA', 'rbx', 'gesta']:
        predicted_classes = ['_'.join(p.split('__')[1].split('_')[:-1]) for p in predicted_trks if 'sub' in p]
    elif method in ['rb', 'finta', 'fiesta-cat', 'fiesta-max-mni', 'fiesta-max-cat']:
        predicted_classes = [p.split('.')[0] for p in predicted_trks if p.endswith('.trk')]
    elif method in ['tractseg']:
        predicted_classes = ['_'.join(p.split('_')[:-1]) for p in predicted_trks if p.endswith('.trk')]
    elif method in ['fiesta-binta']:
        predicted_classes = ['_'.join(p.split('__')[1].split('_')[:-1]) for p in predicted_trks if 'binta_binta' in p]
    else:
        raise NotImplementedError()
    with open('label_int_mappings/TrInF_to_105HCP.json', 'r') as f:
        TrInF_to_105HCP = json.load(f)

    gt_trks = os.listdir(gt)
    gt_classes =['_'.join(g.split('__')[1].split('.')[:-1]) for g in gt_trks if 'sub' in g]
    predicted_paths = [os.path.join(predicted, p) for p in predicted_trks if p.endswith('.trk')]
    gt_paths = [os.path.join(gt, g) for g in gt_trks if 'sub' in g]
    print(len(predicted_classes))
    print(len(gt_classes))

    if 'sub' in sub:
        tractogram = load_tractogram(f"/scratch/anoushkrit.scee.iitmandi/DATA/TractoInferno/derivatives/testset/{sub}/tractography/{sub}__{gt_classes[0]}.trk", "same", bbox_valid_check=False)
    else:
        raise NotImplementedError()
    # if method == 'tractseg':
    gt_classes_hcp = []
    for g in gt_classes:
        if g in TrInF_to_105HCP.keys():
            gt_classes_hcp.append(TrInF_to_105HCP[g])
        else:
            gt_classes_hcp.append('NULL')
    dic = {}
    # cnt = 0
    # cnt2 = 0
    # for trk in gt_paths:
    #     true_streamlines = load_tractogram(trk, "same", bbox_valid_check=False).streamlines
    #     true_streamlines = list(true_streamlines)
    #     cnt2+=(len(true_streamlines))
    for i,trk in enumerate(predicted_paths):
        if predicted_classes[i] not in gt_classes and predicted_classes[i] not in gt_classes_hcp:
            # pred_streamlines = load_tractogram(trk, "same", bbox_valid_check=False).streamlines
            # pred_streamlines = list(pred_streamlines)
            # cnt+=(len(pred_streamlines))
            print('Skipping', predicted_classes[i])
            continue
        if method != 'tractseg':
            j = np.where(np.array(gt_classes) == predicted_classes[i])[0][0]
        else:
            j = np.where(np.array(gt_classes_hcp) == predicted_classes[i])[0][0]
    
        # if predicted_classes[i] == 'MdLF_L':
        #     continue
        print(predicted_classes[i], gt_classes[j])
        true_streamlines = load_tractogram(gt_paths[j], "same", bbox_valid_check=False).streamlines
        pred_streamlines = load_tractogram(trk, "same", bbox_valid_check=False).streamlines
        true_streamlines = list(true_streamlines)
        pred_streamlines = list(pred_streamlines)
        # cnt+=(len(pred_streamlines))
        # plot_3d_streamlines(true_streamlines, f'{sub}__{gt_classes[j]}__true')
        # plot_3d_streamlines(pred_streamlines, f'{sub}__{predicted_classes[i]}__pred')
        t1 = StatefulTractogram.from_sft(ArraySequence(true_streamlines), tractogram)
        t2 = StatefulTractogram.from_sft(ArraySequence(pred_streamlines), tractogram)
        dic[predicted_classes[i]] = hcp_scores(t1, t2)
    # print(cnt,cnt2)
    return [sub, dic]

def calculate_hcp_scores(predicted, gt, save_path, method):
    
    score = calculate_hcp_scores_subject(('sub-1006', gt, predicted, method))
    print(score)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(score[1], f, indent=4)

    fiesta_results = score[1]
    fiesta_scores = []
    dice = {}
    overlap = {}
    overreach = {}
    for class_ in fiesta_results.keys():
        dice[class_] = fiesta_results[class_]['dice']
        overlap[class_] = fiesta_results[class_]['overlap']
        overreach[class_] = fiesta_results[class_]['overreach']

    fiesta_scores.append(dice)
    fiesta_scores.append(overlap)
    fiesta_scores.append(overreach)
    fiesta_df = pd.DataFrame(fiesta_scores, index=['Dice', 'Overlap', 'Overreach'])

    # Sort columns alphabetically
    fiesta_df = fiesta_df.reindex(sorted(fiesta_df.columns), axis=1)
    csv_path = save_path.replace('.json', '.csv')
    fiesta_df.to_csv(csv_path)




if __name__ == "__main__":
    print('Calculating HCP scores')
    # predicted_trk = '/scratch/anoushkrit.scee.iitmandi/DATA/FIESTA/sub1006_results/'
    predicted_trk = '/scratch/anoushkrit.scee.iitmandi/DATA/FIESTA/fiesta-binta-sub1006-results'
    gt_trk = '/scratch/anoushkrit.scee.iitmandi/DATA/TractoInferno/derivatives/testset/sub-1006/tractography/'
    save_path = 'predictions_new/FIESTA/gesta_sub-1006.json'
    calculate_hcp_scores(predicted_trk, gt_trk, save_path, 'gesta')

