from dipy.segment.bundles import ba_analysis
import numpy as np
import pickle
import os
from multiprocessing import Pool
import json
from tqdm import tqdm
import sys


def calculate_buan(args):
    class_path, threshold = args

    pred_streamline_path = os.path.join(class_path, 'pred_streamlines.pkl')
    true_streamline_path = os.path.join(class_path, 'true_streamlines.pkl')
    
    with open(pred_streamline_path, 'rb') as f:
        pred_streamlines = pickle.load(f)
    with open(true_streamline_path, 'rb') as f:
        true_streamlines = pickle.load(f)
        
    pred_streamlines = [np.array(s) for s in pred_streamlines]
    true_streamlines = [np.array(s) for s in true_streamlines]

    ba_score = ba_analysis(pred_streamlines, true_streamlines, threshold=threshold)
    return ba_score, class_path

if __name__ == '__main__':
    
    root_path = 'predictions_new/true_pred_streamlines'
    ba_analysis_save_path = 'predictions_new/ba_analysis3.json'
    methods = os.listdir(root_path)
    method_paths = [os.path.join(root_path, method) for method in methods]

    if os.path.isfile(ba_analysis_save_path):
        with open(ba_analysis_save_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    threshold = 1
    args = []
    for i, method_path in enumerate(method_paths[10:]):
        subjects = os.listdir(method_path)
        subject_paths = [os.path.join(method_path, subject) for subject in subjects if subject in ['599469', 'sub-1006']]

        for subject_path in subject_paths:
            classes = os.listdir(subject_path)
            class_paths = [os.path.join(subject_path, class_) for class_ in classes]

            for class_path in class_paths:
                args.append([class_path, threshold])
    
    print('input args:', len(args))

    with Pool(20) as p:
        ba_scores = list(tqdm(p.imap(calculate_buan, args), total=len(args)))

    for ba_score, class_path in ba_scores:
        method = class_path.split('/')[-3]
        subject = class_path.split('/')[-2]
        class_ = class_path.split('/')[-1]

        results[method][subject][class_] = ba_score    
    
    os.makedirs(os.path.dirname(ba_analysis_save_path), exist_ok=True)
    with open(ba_analysis_save_path, 'w') as f:
        json.dump(results, f, indent=4)
