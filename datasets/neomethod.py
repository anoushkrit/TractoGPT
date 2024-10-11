# %%
import sys
import os,json
import datetime
import numpy as np
from pathlib import Path
import torch
from blist import blist
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.interpolate import CubicSpline
from multiprocessing import Pool, cpu_count, set_start_method
# from dipy.tracking.streamline import select_random_set_of_streamlines, set_number_of_points
sys.path.append('/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/')
from utils.config import *
import gc, pickle
import copy
from memory_profiler import profile
from .build import DATASETS

#%%
def create_clusters_subject(args):
    data, subject_id = args
    clusters = []
    cluster_labels = []
    included_streamlines = np.zeros(len(data['label'])) # none included here
    cluster_ids_l3, cluster_count_l3 = np.unique(data['cluster_ids_l3'], return_counts=True)

    cluster_ids_l2 = []
    for cluster_id in cluster_ids_l3[cluster_count_l3 < 10]:
        cluster_ids_l2.append(data['cluster_ids_l2'][np.where(data['cluster_ids_l3'] == cluster_id)[0][0]])
    cluster_ids_l2 = np.unique(cluster_ids_l2)
    cluster_count_l2 = np.zeros(len(cluster_ids_l2))
    for i,cluster_id in enumerate(cluster_ids_l2):
        cluster_count_l2[i] = len(np.where(data['cluster_ids_l2'] == cluster_id)[0])
    
    cluster_ids_l1 = []
    for cluster_id in cluster_ids_l2[cluster_count_l2 < 10]:
        cluster_ids_l1.append(data['cluster_ids_l1'][np.where(data['cluster_ids_l2'] == cluster_id)[0][0]])
    cluster_ids_l1 = np.unique(cluster_ids_l1)
    cluster_count_l1 = np.zeros(len(cluster_ids_l1))
    for i,cluster_id in enumerate(cluster_ids_l1):
        cluster_count_l1[i] = len(np.where(data['cluster_ids_l1'] == cluster_id)[0])
    
    for cluster_id in cluster_ids_l1:
        cluster_streamline_idx = np.where(data['cluster_ids_l1'] == cluster_id)[0]
        if included_streamlines[cluster_streamline_idx[0]] == 1:
            continue
        if len(cluster_streamline_idx)<10:
            continue
        included_streamlines[cluster_streamline_idx] = 1
        u_labels, u_counts = np.unique(data['label'][cluster_streamline_idx], return_counts=True)
        label = u_labels[np.argmax(u_counts)]
        cluster_streamline_idx = [f'{subject_id}_{i}' for i in cluster_streamline_idx]
        clusters.append(cluster_streamline_idx)
        cluster_labels.append(label)

    for cluster_id in cluster_ids_l2:
        cluster_streamline_idx = np.where(data['cluster_ids_l2'] == cluster_id)[0]
        if included_streamlines[cluster_streamline_idx[0]] == 1:
            continue
        if len(cluster_streamline_idx)<10:
            continue
        included_streamlines[cluster_streamline_idx] = 1
        u_labels, u_counts = np.unique(data['label'][cluster_streamline_idx], return_counts=True)
        label = u_labels[np.argmax(u_counts)]
        cluster_streamline_idx = [f'{subject_id}_{i}' for i in cluster_streamline_idx]
        clusters.append(cluster_streamline_idx)
        cluster_labels.append(label)

    for cluster_id in cluster_ids_l3:
        cluster_streamline_idx = np.where(data['cluster_ids_l3'] == cluster_id)[0]
        if included_streamlines[cluster_streamline_idx[0]] == 1:
            continue
        if len(cluster_streamline_idx)<10:
            continue
        included_streamlines[cluster_streamline_idx] = 1
        u_labels, u_counts = np.unique(data['label'][cluster_streamline_idx], return_counts=True)
        label = u_labels[np.argmax(u_counts)]
        cluster_streamline_idx = [f'{subject_id}_{i}' for i in cluster_streamline_idx]
        clusters.append(cluster_streamline_idx)
        cluster_labels.append(label)

    return (clusters, cluster_labels)
def load_data(args):
    pth, use_resampled_streamline, pts_per_sample = args
    level = 3
    subject_data = {}
    subject_id = os.path.basename(pth).split('.')[0]
    with open(pth, 'rb') as f:
        data = pickle.load(f)
    
    data_x = np.array(data[0])
    subject_data['streamlines'] = blist(data[1])
    subject_data['label'] = np.array(data[2])
    subject_data['cluster_ids_l1'] = data_x[:,level-2]
    subject_data['cluster_ids_l2'] = data_x[:,level-1]
    subject_data['cluster_ids_l3'] = data_x[:,level]

    cluster, cluster_labels = create_clusters_subject((subject_data, subject_id))

    cluster_pc = np.zeros((len(cluster), pts_per_sample, 3))
    for i in range(len(cluster)):
        cluster_pc[i] = concat_streamlines((subject_data, cluster[i], use_resampled_streamline, pts_per_sample))
    return (cluster_pc, cluster_labels, cluster)

def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
def concat_streamlines(args):
    data, indices, use_resampled_streamline, pts_per_sample = args
    subject_id = indices[0].split('_')[0]      
    indices = [int(i.split('_')[1]) for i in indices]
    club_streamlines = np.array(data['streamlines'][indices[0]])  
    for idx in indices[1:]:
        if use_resampled_streamline:
            pass
            # club_streamlines = np.concatenate((club_streamlines, resampled_streamlines[idx]))
        else:
            club_streamlines = np.concatenate((club_streamlines, data['streamlines'][idx]))     
    
    club_streamlines = pc_norm(club_streamlines)
    original_club_streamlines = copy.deepcopy(club_streamlines)
    while len(club_streamlines)+len(original_club_streamlines) < pts_per_sample:
        club_streamlines = np.concatenate((club_streamlines,original_club_streamlines))

    if len(original_club_streamlines) > pts_per_sample:
        club_streamlines = original_club_streamlines[np.random.choice(len(original_club_streamlines), pts_per_sample, replace=False)]
    else:
        club_streamlines = np.concatenate((club_streamlines,original_club_streamlines[np.random.choice(len(original_club_streamlines), pts_per_sample-len(club_streamlines), replace=False)]))
    
    return np.array(club_streamlines)
@DATASETS.register_module()
class neomethod(Dataset):

    def __init__(self,config):
        self.config = config
        self.data_path = config.data_path
        self.subset = config.subset
        self.save_path = os.path.join(self.data_path, 'processed_data')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.data_path = os.path.join(self.data_path, self.subset)
        self.use_custom_batch = config.use_custom_batch
        self.pts_per_sample = config.pts_per_sample
        self.min_batches = config.minimum_batches
        self.samples_per_class_per_batch = config.samples_per_class_per_batch
        self.use_resampled_streamline = config.use_resampled_streamline
        self.min_threshold_per_cluster = config.min_threshold_per_cluster
        self.label_mapping_path = config.label_mapping_path
        self.level = config.level
        if self.subset == 'validset':
            self.use_custom_batch = False

        if self.use_resampled_streamline:
            raise NotImplementedError('Resampled Streamlines not implemented yet')

        self.label_int_mapping = self.load_label_mapping()
        print(datetime.datetime.now())
        if os.path.exists(os.path.join(self.save_path, f'{self.subset}.pkl')):
            print(os.path.join(self.save_path, f'{self.subset}.pkl'))
            with open(os.path.join(self.save_path, f'{self.subset}.pkl'), 'rb') as f:
                self.cluster_pc, self.cluster_labels = pickle.load(f)
        else:
            self.load_and_process_data()
        print(datetime.datetime.now())
        self.create_custom_batches()
        print(datetime.datetime.now())
    
    def load_label_mapping(self):
        with open(self.label_mapping_path, 'r') as f:
            label_mapping = json.load(f)
        
        return label_mapping

    def create_custom_batches(self):
        unique_labels, label_counts = np.unique(self.cluster_labels, return_counts=True)
        print('min_label: ',min(label_counts))
        batch_size = self.samples_per_class_per_batch*len(unique_labels)
        num_batches = max(int(min(label_counts)//self.samples_per_class_per_batch), self.min_batches)
        self.custom_batches = np.zeros((num_batches, batch_size), dtype=int)

        for i, label in enumerate(unique_labels):
            label_indices = np.where(self.cluster_labels == label)[0]
            num_samples = self.samples_per_class_per_batch*num_batches
            full_rounds = num_samples//len(label_indices)
            remainder = num_samples%len(label_indices)
            new_label_indices = np.tile(label_indices, full_rounds)
            if remainder > 0:
                new_label_indices = np.concatenate((new_label_indices, np.random.choice(label_indices, remainder, replace=False)))
            np.random.shuffle(label_indices)

            label_indices = new_label_indices.reshape((num_batches, self.samples_per_class_per_batch))
            self.custom_batches[:, i*self.samples_per_class_per_batch:(i+1)*self.samples_per_class_per_batch] = label_indices

    def load_and_process_data(self):

        subject_ids = os.listdir(self.data_path)
        inp = [(os.path.join(self.data_path,i), self.use_resampled_streamline, self.pts_per_sample) for i in subject_ids]
        with Pool() as p:
            print('inside pool')
            data = p.map(load_data, inp)
        
        self.cluster_pc = []
        self.cluster_labels = []
        self.cluster = []
        for i in range(len(data)):
            self.cluster_pc.extend(data[i][0])
            self.cluster_labels.extend(data[i][1])
            self.cluster.extend(data[i][2])
        
        self.cluster_pc = np.array(self.cluster_pc)
        self.cluster_labels = np.array(self.cluster_labels)
        
        with open(os.path.join(self.save_path, f'{self.subset}.pkl'), 'wb') as f:
            pickle.dump((self.cluster_pc, self.cluster_labels), f)
        del data
        gc.collect()

    def map_batch_labels(self, labels):
        int_labels = [int(self.label_int_mapping[class_label]) for class_label in labels]
        return np.array(int_labels)
    
    def __getitem__(self, index):
        if not self.use_custom_batch:
            label = self.map_batch_labels([self.cluster_labels[index]])
            pc = self.cluster_pc[index]
            return torch.tensor(label).long(), [0], torch.tensor(pc).float()
        else:
            batch_label = self.map_batch_labels([self.cluster_labels[i] for i in self.custom_batches[index]])
            batch_pc = np.array([self.cluster_pc[i] for i in self.custom_batches[index]])

            shuffle_batch = np.random.choice(len(batch_label), len(batch_label), replace=False)
            batch_label = batch_label[shuffle_batch]
            batch_pc = batch_pc[shuffle_batch]

            return torch.from_numpy(batch_label.astype(np.float32)), [0], torch.from_numpy(batch_pc.astype(np.float32))
    def __len__(self):
        if not self.use_custom_batch:
            return len(self.cluster_pc)
        else:
            return len(self.custom_batches)
        
#%%
# config = cfg_from_yaml_file('/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/cfgs/dataset_configs/neomethod.yaml')
# dataset = neomethod(config)
# #%%
# cnt = 0
# for i in range(len(dataset.cluster)):
#     if(len(dataset.cluster[i]))<10:
#         cnt+=1
# print(cnt)
# print(len(np.unique(dataset.cluster_labels, return_counts=True)[1]))
# import plotly.graph_objects as go

# def plot_point_cloud(points, labels):
#     x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
#     # Create a 3D scatter plot
#     fig = go.Figure(data=[go.Scatter3d(
#         x=x, y=y, z=z, mode='markers',
#         marker=dict(size=3, color=labels, colorscale='Viridis', colorbar=dict(title="Label")),
#     )])

#     # Add layout settings
#     fig.update_layout(scene=dict(
#         xaxis_title='X',
#         yaxis_title='Y',
#         zaxis_title='Z'),
#         title=f'3D Point Cloud with {labels}',
#         autosize=True
#     )

#     fig.show()
# #%%
# # print(len(dataset))
# idx = 89432
# plot_point_cloud(dataset[idx][2], dataset[idx][0])
# # #%%
# # cluster, cluster_label = dataset[0]
# # x = []
# # for i in cluster:
# #     x.append(len(i))

# # x,y = np.unique(x, return_counts=True)
# # import matplotlib.pyplot as plt
# # plt.scatter(x,y)
# # #%%
# # max(x)
# # #%%
# # import pickle
# # import numpy as np
# # with open('/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/data/dataTrInF/qbx/validset/sub-1254.pkl', 'rb') as f:
# #     data = pickle.load(f)
# # z = np.array(data[0])[:,3]
# # #%%
# # a,b = np.unique(z, return_counts=True)
# # max(b)
# # #%%
# # dataset[10]
# #%%
# print(dataset.custom_batches.shape)