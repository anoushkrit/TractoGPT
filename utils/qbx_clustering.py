#%%
import numpy as np
from dipy.io.streamline import load_trk
from nibabel.streamlines.array_sequence import ArraySequence
from dipy.segment.clustering import QuickBundlesX
# https://docs.dipy.org/stable/reference/dipy.segment.html#centerofmassfeature
from pathlib import Path
from collections import Counter
import os , pandas as pd
from tqdm import tqdm
import sys
sys.path.append('/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/utils')
from streamline_ops import *
from sklearn.utils.class_weight import compute_class_weight
from dipy.tracking.distances import approx_polygon_track
from tqdm import tqdm
import multiprocessing
import pickle



#%%
is_interactive = True

def find_duplicates(sub_arr):
    occurrences = {}
    duplicate_indices = []
    for idx, item in enumerate(sub_arr):
        arr_tuple = str(item)  # Convert item to string for hashing
        if arr_tuple in occurrences:
            duplicate_indices.append(idx)  # Store index of duplicate
        else:
            occurrences[arr_tuple] = idx  # Store first occurrence

    del occurrences
    return duplicate_indices

def chunkify(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def deduplication(arr, labels):
    # arr = tracks
    num_chunks = multiprocessing.cpu_count()
    chunks = list(chunkify(arr, len(arr) // num_chunks))

    with multiprocessing.Pool(processes=num_chunks) as pool:
        results = pool.map(find_duplicates, chunks)
    total_duplicate_indices = [index for sublist in results for index in sublist]
    arr = [arr[i] for i in range(len(arr)) if i not in total_duplicate_indices]
    labels = [labels[i] for i in range(len(labels)) if i not in total_duplicate_indices]
    print(len(arr), "after deduplication")
    return arr, labels

def load_TractoInferno(tracts_directory):
    tracks = [] 
    labels = [] 
    import re        
    paths = sorted([p for p in Path(tracts_directory).iterdir() if p.is_file() and p.suffix == '.trk'])
    # for path in tqdm(paths):
    for path in paths:
        subject = re.search(r'\d{4}', str(path)).group()
        # print(subject)
        if path.is_file() and path.suffix == '.trk':
            label = re.search(r'__(.*?)\.trk', str(path)).group(1)
            tractogram = load_trk(str(path), "same", bbox_valid_check=False)
            num_streamlines = len(tractogram.streamlines)
        else:
            raise Warning(f"Warning: {path} is not a trk file, skipping")
        with open(os.path.join(tracts_directory, f"tractoinferno_{subject}.txt"), "a") as f:
            f.write(f"Number of Streamline in {label} is {num_streamlines}\n")
        for s in tractogram.streamlines:
            labels.append(label)
            tracks.append(s)
    # print(F"Removing Duplicates from {len(tracks)} streamlines")
    tracks, labels = deduplication(tracks, labels)
    return tracks, labels

def load_HCP(tracts_directory):
    tracks = [] 
    labels = [] 
    import re
    dataset = "HCP"
    subject = re.search(r'\d{6}', tracts_directory).group()
    print(f"Loading streamlines for {subject} for {dataset} dataset")
    paths = sorted([p for p in Path(tracts_directory).iterdir() if p.is_file() and p.suffix == '.trk'])
    for path in tqdm(paths):
        if path.is_file() and path.suffix == '.trk':
            label = str(path).split('/')[-1].split('.')[0]
            tractogram = load_trk(str(path), "same", bbox_valid_check=False)
            num_streamlines = len(tractogram.streamlines)
        else:
            raise Warning(f"Warning: {path} is not a trk file, skipping")
        with open(os.path.join(tracts_directory, f"{dataset}_{subject}.txt"), "a") as f:
            f.write(f"Number of Streamline in {label} is {num_streamlines}\n")
        for s in tractogram.streamlines:
            labels.append(label)
            tracks.append(s)
    return tracks, labels

def stratified_random_sample(tracks, labels, shuffle=True):
    tracks = tracks 
    labels = labels
    n_tracks = len(tracks)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    random_indices = []
    label_counts = Counter(labels)
    n_samples = np.percentile(list(label_counts.values()), 20)
    random_indices = []
    print("Performing Stratified Random Sampling")
    for label in tqdm(np.unique(labels)):
        label_indices = np.where(np.array(labels) == label)[0]
        # print(label, len(label_indices))
        len_rand = len(random_indices)
        if n_samples < len(label_indices):
            random_indices.extend(list(np.random.RandomState(42).permutation(label_indices))[:int(n_samples)])
        else: 
            random_indices.extend(list(np.random.RandomState(42).permutation(label_indices))[:])
        # print(f"Random Indices Length: {len(random_indices)}")
        # print(f"Difference of Random Indices: {len(random_indices) - len_rand}")
    if shuffle == True:
        np.random.shuffle(random_indices)
    tracks = [tracks[i] for i in random_indices]
    labels = [labels[i] for i in random_indices]
    # tracks, labels = stratified_random_sample(original_tracks, original_labels)
    return tracks, labels

#%%
def qbx_cci_clustering(tracks, labels, qbx_hierarchy, downsample=True):
    downsample = True
    if type(tracks) == list: 
        tracks = ArraySequence(tracks)
    if downsample == True: 
        tracks = [approx_polygon_track(t, 0.5) for t in tracks]
    # print("Num Tracks: ", len(tracks), "| Clustering Streamlines")
    lvl = len(qbx_hierarchy)
    qb = QuickBundlesX(qbx_hierarchy, metric = 'MDF_12points')
    clusters = qb.cluster(tracks)
    return clusters

def make_tracks_clusters_map(clusters, cidx):
    result = [] 
    for cluster_index, cluster in enumerate(list(clusters.get_clusters(cidx))):
        for track_index in cluster:
            result.append((track_index, cluster_index))

    result.sort(key=lambda x: x[0])
    track_indices, cluster_indices = zip(*result)    
    return track_indices, cluster_indices

qbx_hierarchy = [40,30,20,10,8,6,4]

level = len(qbx_hierarchy)

#%%
def func(args):
    # pth = f'/scratch/anoushkrit.scee.iitmandi/DATA/TractoInferno/derivatives/{split}/{subject}/tractography'
    pth, split, subject = args
    tracks, labels = load_TractoInferno(tracts_directory=pth)
    clusters = qbx_cci_clustering(tracks, labels, qbx_hierarchy, downsample=True)
    tidxs_4, cidxs_4 = make_tracks_clusters_map(clusters, level-2)
    tidxs_5, cidxs_5 = make_tracks_clusters_map(clusters, level-1)
    tidxs_6, cidxs_6 = make_tracks_clusters_map(clusters, level)

    df_4 = pd.DataFrame({'tidxs': tidxs_4, 'cidxs': cidxs_4})
    df_5 = pd.DataFrame({'tidxs': tidxs_5, 'cidxs': cidxs_5})
    df_6 = pd.DataFrame({'tidxs': tidxs_6, 'cidxs': cidxs_6})

    merged_df = pd.merge(df_4, df_5, on='tidxs', suffixes=('_level_4', '_level_5'))

    # Merge the result with df_6
    final_merged_df = pd.merge(merged_df, df_6, on='tidxs', suffixes=('', '_level_6'))

    final_merged_df.columns = ['track_indices', '3last', '2last', 'last']
    final_list = final_merged_df.values.tolist()

    with open(f"/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/data/dataTrInF/qbx/{split}/{subject}.pkl", "wb") as f:
        pickle.dump((final_list, tracks, labels), f)

for split in ["trainset", "testset", "validset"]:
    print(split)
    if not os.path.exists(f"/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/data/dataTrInF/qbx/{split}"):
        os.makedirs(f"/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/data/dataTrInF/qbx/{split}")
    for subject in tqdm(os.listdir(f"/scratch/anoushkrit.scee.iitmandi/DATA/TractoInferno/derivatives/{split}")):
        pth = f'/scratch/anoushkrit.scee.iitmandi/DATA/TractoInferno/derivatives/{split}/{subject}/tractography'
        tracks, labels = load_TractoInferno(tracts_directory=pth)
        clusters = qbx_cci_clustering(tracks, labels, qbx_hierarchy, downsample=True)
        tidxs_4, cidxs_4 = make_tracks_clusters_map(clusters, level-2)
        tidxs_5, cidxs_5 = make_tracks_clusters_map(clusters, level-1)
        tidxs_6, cidxs_6 = make_tracks_clusters_map(clusters, level)

        df_4 = pd.DataFrame({'tidxs': tidxs_4, 'cidxs': cidxs_4})
        df_5 = pd.DataFrame({'tidxs': tidxs_5, 'cidxs': cidxs_5})
        df_6 = pd.DataFrame({'tidxs': tidxs_6, 'cidxs': cidxs_6})

        merged_df = pd.merge(df_4, df_5, on='tidxs', suffixes=('_level_4', '_level_5'))

        # Merge the result with df_6
        final_merged_df = pd.merge(merged_df, df_6, on='tidxs', suffixes=('', '_level_6'))

        final_merged_df.columns = ['track_indices', '3last', '2last', 'last']
        final_list = final_merged_df.values.tolist()

        print('saving file at', f"/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/data/dataTrInF/qbx/{split}/{subject}.pkl")
        with open(f"/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/data/dataTrInF/qbx/{split}/{subject}.pkl", "wb") as f:
            pickle.dump((final_list, tracks, labels), f)

