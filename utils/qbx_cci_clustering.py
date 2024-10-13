#%%
import numpy as np
from nibabel.streamlines.array_sequence import ArraySequence
from dipy.segment.clustering import QuickBundlesX
# https://docs.dipy.org/stable/reference/dipy.segment.html#centerofmassfeature
from pathlib import Path
from collections import Counter
import os 
from tqdm import tqdm
from streamline_ops import *
from sklearn.utils.class_weight import compute_class_weight
from dipy.tracking.distances import approx_polygon_track
from tqdm import tqdm
import multiprocessing
from dipy.tracking.streamline import cluster_confidence
from dice import hcp_scores
from dipy.io.streamline import load_tractogram   
import pickle
import re 

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
    if tracts_directory.endswith('.trk'):
        path = tracts_directory
        if not tracts_directory.endswith('.trk'):
            raise Warning(f"Warning: {path} is not a trk file, skipping")
        subject = re.search(r'\d{4}', str(path)).group()
        label = re.search(r'__(.*?)\.trk', str(path)).group(1)
        tractogram = load_tractogram(str(path), "same", bbox_valid_check=False)
        num_streamlines = len(tractogram.streamlines)
        for s in tractogram.streamlines:
            labels.append(label)
            tracks.append(s)
        print(F"Removing Duplicates from {len(tracks)} streamlines")
        tracks, labels = deduplication(tracks, labels)
        return tracks, labels, tractogram, subject
    else:
        paths = sorted([p for p in Path(tracts_directory).iterdir() if p.is_file() and p.suffix == '.trk'])
        for path in tqdm(paths):
            subject = re.search(r'\d{4}', str(path)).group()
            # print(subject)
            if path.is_file() and path.suffix == '.trk':
                label = re.search(r'__(.*?)\.trk', str(path)).group(1)
                tractogram = load_tractogram(str(path), "same", bbox_valid_check=False)
                num_streamlines = len(tractogram.streamlines)
            else:
                raise Warning(f"Warning: {path} is not a trk file, skipping")
            with open(os.path.join(tracts_directory, f"tractoinferno_{subject}.txt"), "a") as f:
                f.write(f"Number of Streamline in {label} is {num_streamlines}\n")
            for s in tractogram.streamlines:
                labels.append(label)
                tracks.append(s)
        print(F"Removing Duplicates from {len(tracks)} streamlines")
        tracks, labels = deduplication(tracks, labels)
        return tracks, labels, tractogram, subject


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
            tractogram = load_tractogram(str(path), "same", bbox_valid_check=False)
            num_streamlines = len(tractogram.streamlines)
        else:
            raise Warning(f"Warning: {path} is not a trk file, skipping")
        with open(os.path.join(tracts_directory, f"{dataset}_{subject}.txt"), "a") as f:
            f.write(f"Number of Streamline in {label} is {num_streamlines}\n")
        for s in tractogram.streamlines:
            labels.append(label)
            tracks.append(s)
    print(F"Removing Duplicates from {len(tracks)} streamlines")
    tracks, labels = deduplication(tracks, labels)
    return tracks, labels, tractogram, subject

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
        len_rand = len(random_indices)
        if n_samples < len(label_indices):
            random_indices.extend(list(np.random.RandomState(42).permutation(label_indices))[:int(n_samples)])
        else: 
            random_indices.extend(list(np.random.RandomState(42).permutation(label_indices))[:])
    if shuffle == True:
        np.random.shuffle(random_indices)
    tracks = [tracks[i] for i in random_indices]
    labels = [labels[i] for i in random_indices]
    # tracks, labels = stratified_random_sample(original_tracks, original_labels)
    return tracks, labels
#%%
def qbx_clustering(tracks, labels, qbx_hierarchy, downsample=True):
    downsample = True
    if type(tracks) == list: 
        tracks = ArraySequence(tracks)
    if downsample == True: 
        tracks = [approx_polygon_track(t, 0.5) for t in tracks]
    print("Num Tracks: ", len(tracks), "| Clustering Streamlines")
    # lvl = len(qbx_hierarchy)
    qb = QuickBundlesX(qbx_hierarchy, metric = 'MDF_12points')
    clusters = qb.cluster(tracks)
    # cluster  = list(clusters.get_clusters(lvl)[0])
    print("Done with Clustering")

    return clusters

def make_tracks_clusters_map(clusters, cidx):
    result = [] 
    for cluster_index, cluster in enumerate(list(clusters.get_clusters(cidx))):
        for track_index in cluster:
            result.append((track_index, cluster_index))

    result.sort(key=lambda x: x[0])
    track_indices, cluster_indices = zip(*result)    
    return track_indices, cluster_indices


#%%
def visualise_cci(select_cci, select_tracks): 
    scene = window.Scene()

    hue = [0.5, 1]
    saturation = [0.0, 1.0]

    lut_cmap = actor.colormap_lookup_table(scale_range=(select_cci.min(), select_cci.max()/4),
                                        hue_range=hue,
                                        saturation_range=saturation)

    bar3 = actor.scalar_bar(lut_cmap)
    scene.add(bar3)

    stream_actor = actor.line(select_tracks, select_cci, linewidth=0.1,
                            lookup_colormap=lut_cmap)
    scene.add(stream_actor)
    interactive = True
    if interactive:
        window.show(scene)
    window.record(scene, n_frames=1, out_path='cci_streamlines.png',
                size=(800, 800))
    
def remove_small_clusters(clusters, level, threshold =5):
    big_clusters = clusters.get_clusters(level) > threshold
    big_cluster_indices = np.where(big_clusters)[0]
    return big_cluster_indices

def cci_filter(cluster, cluster_indices, labels, cluster_label = None, val = False, viz = False): 
    """Here cluster is a list of streamline tracks"""
    
    cci_mdf = cluster_confidence(cluster, max_mdf=5, subsample=16, power=2, override=False)
    cci_threshold =  np.percentile(cci_mdf,  5)
    cci_bool = cci_mdf > cci_threshold 
    select_cci_indices = np.where(cci_bool)[0]
    select_cluster_indices = cluster_indices[select_cci_indices.tolist()]
    # assert len(select_cluster_indices) == len(cluster_indices)
    # print(select_cluster_indices)
    if val == True:
        if cluster_label is None:
            raise Warning("Please provide cluster label")
        print("Before CCI", dict(Counter(cluster_label)), len(cluster_indices))
        select_labels = get_values(select_cluster_indices, labels)
        print("After CCI" , dict(Counter(select_labels)), len(select_cluster_indices))
        if viz: 
            visualise_cci(cci_mdf, cluster)
    

class QualityControl:
    def __init__(self, clusters, labels, level ): 
        super(QualityControl, self).__init__()
        self.clusters = clusters
        self.level = level

    def cluster_stats(self, cidx, original_labels): 
        streamlines_in_cluster = list(self.clusters.get_clusters(self.level)[cidx])
        # select_tracks = [original_tracks[t] for t in streamlines_in_cluster]
        stat_labels = [original_labels[l] for l in streamlines_in_cluster]
        return len(streamlines_in_cluster), dict(Counter(stat_labels))

    def check_label_proportion(self): 
        for le in range(len(self.clusters.get_clusters(self.level))):
            lenth, dict_streamlines = self.cluster_stats(le, self.labels, self.level)
            print(le, lenth, dict_streamlines)
        return

    def dice_score(): 
        return
    
get_values = lambda indices, query: [query[i] for i in indices]

def qbx_cci_clustering(subject_directory: str, 
                       dataset: str,
                       qbx_hierarchy: list,
                       small_cluster_threshold: int,
                       save_path: str,
                       return_list: bool = False,
                       level: int = None,
                       downsample: bool = True):
    if dataset == "HCP":
        data_string = "hcp"
        tracks, labels, tractogram, sub = load_HCP(tracts_directory=subject_directory)
    elif dataset == "TractoInferno":
        data_string = "tf"
        tracks, labels, tractogram, sub = load_TractoInferno(tracts_directory=subject_directory)
    else: 
        raise ValueError("Dataset should be one of HCP or TractoInferno")
    clusters = qbx_clustering(tracks, labels, qbx_hierarchy, downsample=downsample)
    if level is None:
        level = len(qbx_hierarchy)
    len_clusters = len(clusters.get_clusters(level))
    print(f"Segmented {len_clusters} for subject {sub} at level {level} using QuickBundlesX at {qbx_hierarchy}")
    big_cluster_indices = remove_small_clusters(clusters, threshold=small_cluster_threshold, level = level)
    print(f"After removing {len_clusters- len(big_cluster_indices)} small clusters, less than {small_cluster_threshold}, we have {len(big_cluster_indices)}")
    idx_clu_label = []
    for nc, bc in tqdm(enumerate(big_cluster_indices)):
        cluster_indices = clusters.get_clusters(level)[bc]
        lci = len(cluster_indices)
        cluster = get_values(cluster_indices, tracks)
        cluster_label  = get_values(cluster_indices, labels)
        ik_list = [nc] * lci
        bc_list = [bc] * lci
        sub_list = [sub] * lci
        idx_clu_label.append((ik_list, bc_list, list(cluster_indices), cluster, cluster_label, sub_list))

    if save_path is not None:

        if os.path.isdir(subject_directory):
            with open(f"{save_path}/{data_string}_{sub}.pkl", "wb") as f:
                pickle.dump(idx_clu_label, f)
            print(f"pickle created at {save_path}/{data_string}_{sub}.pkl")
        else:
            label = re.search(r'__(.*?)\.trk', str(subject_directory)).group(1)
            with open(f"{save_path}/{data_string}_{sub}_{label}.pkl", "wb") as f:
                pickle.dump(idx_clu_label, f)
            print(f"pickle created at {save_path}/{data_string}_{sub}.pkl")


        return idx_clu_label
    else:
        if return_list is False: 
            return idx_clu_label
        else: 
            return list(map(list, idx_clu_label))

    # cci_filter(cluster=cluster) # cci filter is not much useful
#%%

if __name__ == "__main__":
    split = "train"
    sub = "1030"
    tracts_path = f"/scratch/anoushkrit.scee.iitmandi/DATA/TractoInferno/derivatives/{split}set/sub-{sub}/tractography"
    # TODO: add func to handle a single trk file 
    idx_clu_label = qbx_cci_clustering(subject_directory= tracts_path, 
                       dataset = "TractoInferno",
                       qbx_hierarchy = [40,30,20,10,8,6,4],
                       small_cluster_threshold = 5,
                       save_path = "/s3/DATA/ClusterData",
                       return_list = False,
                       level =None,
                       downsample= True)

    # ik_list, bc_list, list(cluster_indices), cluster, cluster_label, sub_list = idx_clu_label[0]
    #%%
    cl1 = idx_clu_label[0][3]
    cl2 = idx_clu_label[1][3]
    cl1_label = idx_clu_label[0][4]
    cl2_label = idx_clu_label[1][4]

    #%%

    # visualize_tracks(cl1, cl2)
    #%%
    # tractogram is subject specific, and DICE is also subject specific. 
    # When calculating DICE, we need to use the same tractogram.for the same subject
    tractogram = load_tractogram(f"/scratch/anoushkrit.scee.iitmandi/DATA/TractoInferno/derivatives/{split}set/sub-{sub}/tractography/sub-{sub}__AF_L.trk", "same", bbox_valid_check=False)
    # DICE, overlap , overreach
    from dipy.io.stateful_tractogram import StatefulTractogram, Space
    from dice import hcp_scores
    t1 = StatefulTractogram(ArraySequence(cl1), tractogram, space=Space.VOX)
    t2 = StatefulTractogram(ArraySequence(cl2), tractogram, space=Space.VOX)
    print(hcp_scores(t1, t2))

    #%%
    # import pandas as pd
    # tidxs_4, cidxs_4 = make_tracks_clusters_map(clusters, level-2)
    # tidxs_5, cidxs_5 = make_tracks_clusters_map(clusters, level-1)
    # tidxs_6, cidxs_6 = make_tracks_clusters_map(clusters, level)
    # df_4 = pd.DataFrame({'tidxs': tidxs_4, 'cidxs': cidxs_4})
    # df_5 = pd.DataFrame({'tidxs': tidxs_5, 'cidxs': cidxs_5})
    # df_6 = pd.DataFrame({'tidxs': tidxs_6, 'cidxs': cidxs_6})
    # merged_df = pd.merge(df_4, df_5, on='tidxs', suffixes=('_level_4', '_level_5'))
    # final_merged_df = pd.merge(merged_df, df_6, on='tidxs', suffixes=('', '_level_6'))
    # final_merged_df.columns = ['track_indices', '3last', '2last', 'last']
    # final_merged_df.to_csv(f"{sub}.csv", index=False)
    # final_list = final_merged_df.values.tolist()
    # paired_list = [[tracks[i], labels[i]] for i in range(len(labels))]
    #%%

    #%%
    import torch 
    import copy 

    split = "test"
    sub = "1047"
    tracts_path = "/s3/DATA/TractoInferno/derivatives/testset/sub-1047/tractography/sub-1047__AF_L.trk"

    #TODO: add func to handle a single trk file
    # TODO: Add upsampling functionlity

    idx_clu_label = qbx_cci_clustering(subject_directory= tracts_path, 
                        dataset = "TractoInferno",
                        qbx_hierarchy = [40,30,20,10,8,6,4],
                        small_cluster_threshold = 5,
                        save_path = "./",
                        return_list = False,
                        level =None,
                        downsample= False)
    pcds = []
    pcds_labels = []

    num_points = 1024
    for i in range(len(idx_clu_label)):
        ik_list, bc_list, cluster_indices, cluster, cluster_label, sub_list = idx_clu_label[i]
        cluster_ = np.concatenate(list(map(np.concatenate, cluster))).reshape(-1,3)
        print(cluster_.shape, len(cluster))
        cluster__= copy.deepcopy(cluster_)
        while cluster_.shape[0] < num_points:
            cluster_ = np.concatenate([cluster_, cluster__])

        cluster_ = cluster_[np.random.choice(cluster_.shape[0], num_points, replace=False)]
        pcds.append(cluster_)
        pcds_labels.append(cluster_label)
    #%%
    pcds = np.array(pcds)
    #%%
    for cla in pcds_labels: 
        print(dict(Counter(cla)))

    #%%
    #%%
    best_model_path = "/home/ang/param-scratch/github/TractoGPT/experiments_TrInf/finetune_single_streamline_TrInF/PointGPT-M/single_streamline_TrInF_150_10000/pps256/ckpt-best.pth"
    model = torch.load_weights(best_model_path)
    pcd = model.predict("")