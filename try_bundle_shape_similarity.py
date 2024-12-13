from dipy.segment.bundles import bundle_shape_similarity
import numpy as np
import pickle
import os
from multiprocessing import Pool
import json
from tqdm import tqdm
import sys
from dipy.io.streamline import load_tractogram


def calculate_buan(args):
    class_path, seed = args
    rng = np.random.default_rng(seed)  # Create RNG using the seed

    # with open('label_int_mappings/TrInF_to_105HCP.json', 'r') as f:
    #     TrInF_to_105HCP = json.load(f)
    # HCP_to_TrInF = {v: k for k, v in TrInF_to_105HCP.items()}
    with open('label_int_mappings/label_int_mapping_105HCP.json', 'r') as f:
        HCP_to_TrInF = json.load(f)
    
    label = class_path.split('/')[-1]
    if label not in HCP_to_TrInF.keys():
        return [None, class_path]


    
    with open(os.path.join(class_path, 'pred_streamlines.pkl'), 'rb') as f:
        pred_streamlines = pickle.load(f)
    
    true_streamlines = list(load_tractogram(f"/scratch/anoushkrit.scee.iitmandi/DATA/dMRI_data/105HCP/test/599469/tracts/{label}.trk", "same", bbox_valid_check=False).streamlines)
    

    # pred_streamline_path = os.path.join(class_path, 'pred_streamlines.pkl')
    # true_streamline_path = os.path.join(class_path, 'true_streamlines.pkl')
    
    # with open(pred_streamline_path, 'rb') as f:
    #     pred_streamlines = pickle.load(f)
    # with open(true_streamline_path, 'rb') as f:
    #     true_streamlines = pickle.load(f)
        
    pred_streamlines = [np.array(s) for s in pred_streamlines]
    true_streamlines = [np.array(s) for s in true_streamlines]

    shape_similarity = bundle_shape_similarity(pred_streamlines, true_streamlines, threshold=10, rng=rng)
    return [shape_similarity, class_path]

if __name__ == '__main__':
    
    root_path = 'predictions_new/true_pred_streamlines'
    buan_save_path = 'predictions_new/buan10.json'
    methods = os.listdir(root_path)
    method_paths = [os.path.join(root_path, method) for method in methods]

    if os.path.isfile('predictions_new/buan.json'):
        with open('predictions_new/buan.json', 'r') as f:
            results = json.load(f)
    else:    
        results = {}

    seed = 0
    args = []

    for i, method_path in enumerate(method_paths):
        subjects = os.listdir(method_path)
        subject_paths = [os.path.join(method_path, subject) for subject in subjects if subject in ['599469', 'sub-1006']]
        if methods[i] in results.keys():
            continue
        for j,subject_path in enumerate(subject_paths):
            classes = os.listdir(subject_path)
            class_paths = [os.path.join(subject_path, class_) for class_ in classes]

            for k,class_path in enumerate(class_paths):
                args.append([class_path, seed])

    print('input args:', len(args))
    
    with Pool() as p:
        shape_similarities = list(tqdm(p.imap(calculate_buan, args), total=len(args)))

    for shape_similarity, class_path in shape_similarities:
        if shape_similarity is None:
            continue
        method = class_path.split('/')[-3]
        subject = class_path.split('/')[-2]
        class_ = class_path.split('/')[-1]
        if method not in results:
            results[method] = {}
        if subject not in results[method]:
            results[method][subject] = {}
        results[method][subject][class_] = shape_similarity
    
    os.makedirs(os.path.dirname(buan_save_path), exist_ok=True)
    with open(buan_save_path, 'w') as f:
        json.dump(results, f, indent=4)