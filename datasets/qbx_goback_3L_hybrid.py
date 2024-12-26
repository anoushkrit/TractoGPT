# Standard libraries
import sys
import os
import json
import datetime
import gc
import pickle
import copy
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count, set_start_method
import shutil

# Third-party libraries
import numpy as np
import torch
from blist import blist
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.interpolate import CubicSpline
from memory_profiler import profile

# Project-specific imports
sys.path.append('./TractoGPT/')
from utils.config import *
from utils.logger import *
from utils.streamline_ops import *
from utils.qbx_clustering import create_qbx_cluster_from_scratch_n_save
from .build import DATASETS


#%%
def load_data(args):
    pth, use_resampled_streamline, pts_per_sample, streamline_threshold_per_cluster, test = args
    level = 3
    subject_data = {}
    subject_id = os.path.basename(pth).split('.')[0]

    with open(pth, 'rb') as f:
        try:
            data = pickle.load(f)
        except:
            return ([], [], [], [], [], [])
    data_x = np.array(data[0])
    subject_data['streamlines'] = data[1]
    subject_data['label'] = np.array(data[2])
    subject_data['cluster_ids_l1'] = data_x[:,level-2]
    subject_data['cluster_ids_l2'] = data_x[:,level-1]
    subject_data['cluster_ids_l3'] = data_x[:,level]
    del data_x
    del data

    cluster, pc_cluster_labels = create_clusters_QBX_3L_per_subject((subject_data, subject_id, streamline_threshold_per_cluster, test))

    pc_cluster = np.zeros((len(cluster), pts_per_sample, 3))
    for i in range(len(cluster)):
        pc_cluster[i] = concat_streamlines((subject_data, cluster[i], use_resampled_streamline, pts_per_sample))
    
    cluster_labels = []
    for i in range(len(cluster)):
        streamline_idx = np.array([int(cluster[i][j].split('_')[1]) for j in range(len(cluster[i]))])
        cluster_labels.append(subject_data['label'][streamline_idx])

    streamlines = subject_data['streamlines']
    subject_labels = subject_data['label']
    del subject_data
    gc.collect()
    if test:
        return (pc_cluster, pc_cluster_labels, cluster, cluster_labels, streamlines, subject_labels)
    return (pc_cluster, pc_cluster_labels, cluster, cluster_labels, None, subject_labels)
    
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
    
    original_club_streamlines = copy.deepcopy(club_streamlines)
    while len(club_streamlines)+len(original_club_streamlines) < pts_per_sample:
        club_streamlines = np.concatenate((club_streamlines,original_club_streamlines))

    if len(original_club_streamlines) > pts_per_sample:
        club_streamlines = original_club_streamlines[np.random.choice(len(original_club_streamlines), pts_per_sample, replace=False)]
    else:
        club_streamlines = np.concatenate((club_streamlines,original_club_streamlines[np.random.choice(len(original_club_streamlines), pts_per_sample-len(club_streamlines), replace=False)]))
    
    return np.array(club_streamlines)
@DATASETS.register_module()
class qbx_goback_3L_hybrid(Dataset):

    def __init__(self,config):
        """ path1 for 105HCP and path2 for TractoInferno """
        self.config = config
        self.data_path = config.data_path
        self.raw_data_path_TrInF = config.raw_data_path_TrInF
        self.raw_data_path_HCP = config.raw_data_path_HCP
        self.subset = config.subset
        self.data_path = os.path.join(self.data_path, self.subset)
        self.save_path = config.save_path

        self.use_custom_batch = config.use_custom_batch
        self.pts_per_sample = config.pts_per_sample
        self.min_batches = config.minimum_batches
        self.samples_per_class_per_batch = config.samples_per_class_per_batch
        self.use_resampled_streamline = config.use_resampled_streamline
        self.min_threshold_per_cluster = config.min_threshold_per_cluster
        self.label_mapping_path = config.label_mapping_path
        self.label_int_mapping = self.load_label_mapping()
        os.makedirs(self.save_path, exist_ok=True)

        if self.subset == 'val' or self.subset == 'test':
            self.use_custom_batch = False

        if self.use_resampled_streamline:
            raise NotImplementedError('Resampled Streamlines not implemented yet')

        if not os.path.exists(self.data_path):
            print_log(f'[DATASET] Data path {self.data_path} does not exist', logger='qbx_goback_3L_hybrid')
            print_log(f'[DATASET] Creating QBX clusters', logger='qbx_goback_3L_hybrid')
            if self.subset in ['train', 'test']:
                TrInF_subset = self.subset+'set'
            else:
                TrInF_subset = self.subset+'idset'
            create_qbx_cluster_from_scratch_n_save(
                dataset='TractoInferno',
                mapping=self.label_int_mapping,
                save_path='/'.join(self.data_path.split('/')[:-1]),
                raw_data_path=self.raw_data_path_TrInF,
                subset=[TrInF_subset]
            )
            create_qbx_cluster_from_scratch_n_save(
                dataset='105HCP',
                mapping=self.label_int_mapping,
                save_path='/'.join(self.data_path.split('/')[:-1]),
                raw_data_path=self.raw_data_path_HCP,
                subset=[self.subset]
            )
        # merge the two datasets
        print_log(f'[DATASET] Merging the two datasets', logger='qbx_goback_3L_hybrid')
        if 'train' in os.listdir('/'.join(self.data_path.split('/')[:-1])) and 'trainset' in os.listdir('/'.join(self.data_path.split('/')[:-1])):
            self.merge_folders(os.path.join('/'.join(self.data_path.split('/')[:-1]), 'train'), os.path.join('/'.join(self.data_path.split('/')[:-1]), 'trainset'))
        if 'val' in os.listdir('/'.join(self.data_path.split('/')[:-1])) and 'validset' in os.listdir('/'.join(self.data_path.split('/')[:-1])):
            self.merge_folders(os.path.join('/'.join(self.data_path.split('/')[:-1]), 'val'), os.path.join('/'.join(self.data_path.split('/')[:-1]), 'validset'))
        if 'test' in os.listdir('/'.join(self.data_path.split('/')[:-1])) and 'testset' in os.listdir('/'.join(self.data_path.split('/')[:-1])):
            self.merge_folders(os.path.join('/'.join(self.data_path.split('/')[:-1]), 'test'), os.path.join('/'.join(self.data_path.split('/')[:-1]), 'testset'))


        if self.subset == 'test' and os.path.exists(os.path.join(self.save_path, f'{self.subset}.pkl')):
            print_log(f'Loading {self.subset} data from {self.save_path}', logger='qbx_goback_3L_hybrid')
            with open(os.path.join(self.save_path, f'{self.subset}.pkl'), 'rb') as f:
                self.pc_cluster, self.pc_cluster_labels, self.cluster, self.cluster_labels, self.streamlines, self.streamline_labels= pickle.load(f)
                for i in self.streamlines.keys():
                    self.streamlines[i] = blist(self.streamlines[i])
            
            if type(self.config.subject_ids) == list:
                self.pc_cluster = [self.pc_cluster[i] for i in range(len(self.pc_cluster)) if self.cluster[i][0].split('_')[0] in self.config.subject_ids]
                self.pc_cluster_labels = [self.pc_cluster_labels[i] for i in range(len(self.pc_cluster_labels)) if self.cluster[i][0].split('_')[0] in self.config.subject_ids]
                self.cluster_labels = [self.cluster_labels[i] for i in range(len(self.cluster_labels)) if self.cluster[i][0].split('_')[0] in self.config.subject_ids]
                self.streamlines = {i: self.streamlines[i] for i in self.streamlines.keys() if i in self.config.subject_ids}
                self.streamline_labels = {i: self.streamline_labels[i] for i in self.streamline_labels.keys() if i in self.config.subject_ids}
                self.cluster = [self.cluster[i] for i in range(len(self.cluster)) if self.cluster[i][0].split('_')[0] in self.config.subject_ids]
                self.pc_cluster = np.array(self.pc_cluster)
                self.pc_cluster_labels = np.array(self.pc_cluster_labels)
        elif os.path.exists(os.path.join(self.save_path, f'{self.subset}.pkl')):
            print_log(f'Loading {self.subset} data from {self.save_path}', logger='qbx_goback_3L_hybrid')
            with open(os.path.join(self.save_path, f'{self.subset}.pkl'), 'rb') as f:
                data = pickle.load(f)
                if len(data) == 4:
                    self.pc_cluster, self.pc_cluster_labels, self.cluster, self.cluster_labels = data
                elif len(data) == 5:
                    self.pc_cluster, self.pc_cluster_labels, self.cluster, self.cluster_labels, _ = data
        else: 
            print_log(f'Loading and Processing {self.subset} data', logger='qbx_goback_3L_hybrid')
            self.load_and_process_data()
        
        print_log(f'Loaded {len(self.pc_cluster)} samples', logger='qbx_goback_3L_hybrid')
        print_log(f'Loaded {len(self.cluster)} clusters', logger='qbx_goback_3L_hybrid')
        print_log(f'[DATASET] Building custom batches', logger='qbx_goback_3L')
        if self.use_custom_batch:
            self.create_custom_batches()

        self.itr_count = 0
    
    def load_label_mapping(self):
        with open(self.label_mapping_path, 'r') as f:
            label_mapping = json.load(f)
        
        return label_mapping
    def merge_folders(self, folder_a, folder_b):
        # Ensure the destination folder exists
        if not os.path.exists(folder_a):
            os.makedirs(folder_a)

        # Iterate through all files and folders in folder_b
        for item in os.listdir(folder_b):
            source_path = os.path.join(folder_b, item)
            destination_path = os.path.join(folder_a, item)

            # If it's a directory, copy recursively
            if os.path.isdir(source_path):
                shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
            else:
                # If it's a file, copy it
                shutil.copy2(source_path, destination_path)
        
        # delete the folder_b
        shutil.rmtree(folder_b)
        print_log(f"Contents of '{folder_b}' have been merged into '{folder_a}'.", logger='qbx_goback_3L_hybrid')
        gc.collect()
    def create_custom_batches(self):
        unique_labels, label_counts = np.unique(self.pc_cluster_labels, return_counts=True)
        batch_size = self.samples_per_class_per_batch*len(unique_labels)
        num_batches = max(int(min(label_counts)//self.samples_per_class_per_batch), self.min_batches)
        self.custom_batches = np.zeros((num_batches, batch_size), dtype=int)

        for i, label in enumerate(unique_labels):
            label_indices = np.where(self.pc_cluster_labels == label)[0]
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

        subjects = [subject for subject in os.listdir(self.data_path) if subject.endswith('.pkl')]
        if self.subset == 'test' and type(self.config.subject_ids) == list: # using only the given subjects
            subjects = [subject for subject in subjects if subject.split('.')[0] in self.config.subject_ids]
        inp = []
        for subject in subjects:
            inp.append((os.path.join(self.data_path,subject), self.use_resampled_streamline, self.pts_per_sample, self.min_threshold_per_cluster, self.subset == 'test'))
        with Pool() as p:
            # data = p.map(load_data, inp)
            data = list(tqdm(p.imap(load_data, inp), total=len(inp)))
        
        self.pc_cluster = []
        self.pc_cluster_labels = []
        self.cluster = []
        self.cluster_labels = []
        self.streamlines = {}
        self.streamline_labels = {}
        for i in range(len(data)):
            self.pc_cluster.extend(data[i][0])
            self.pc_cluster_labels.extend(data[i][1])
            self.cluster.extend(data[i][2])
            self.cluster_labels.extend(data[i][3])
            self.streamline_labels[subjects[i].split('.')[0]] = data[i][5]
            if self.subset == 'test':
                self.streamlines[subjects[i].split('.')[0]] = data[i][4]
        
        self.pc_cluster = np.array(self.pc_cluster)
        self.pc_cluster_labels = np.array(self.pc_cluster_labels)
        
        os.makedirs(self.save_path, exist_ok=True)
        if self.subset != 'test':
            with open(os.path.join(self.save_path, f'{self.subset}.pkl'), 'wb') as f:
                pickle.dump((self.pc_cluster, self.pc_cluster_labels, self.cluster, self.cluster_labels), f)
        # else:
        #     with open(os.path.join(self.save_path, f'{self.subset}.pkl'), 'wb') as f:
        #         pickle.dump((self.pc_cluster, self.pc_cluster_labels, self.cluster, self.cluster_labels, self.streamlines, self.streamline_labels), f)
        del data
        gc.collect()

    def map_batch_labels(self, labels):
        int_labels = [int(self.label_int_mapping[class_label]) for class_label in labels]
        return np.array(int_labels)
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    def __getitem__(self, index):
        if self.subset == 'test':
            label = self.map_batch_labels([self.pc_cluster_labels[index]])
            pc = self.pc_cluster[index]
            pc = self.pc_norm(pc)
            subject = self.cluster[index][0].split('_')[0]
            streamline_idx = [int(i.split('_')[1]) for i in self.cluster[index]]
            streamlines = [list(self.streamlines[subject][i]) for i in streamline_idx]
            streamline_labels = self.streamline_labels[subject][streamline_idx]
            streamline_labels = list(self.map_batch_labels(streamline_labels))
            subject = [subject]*len(streamlines)
            return torch.tensor(label).long(), [str(subject), str(streamline_labels), str(streamlines)], torch.tensor(pc).float()
            # return torch.tensor(label).long(), [str(0), str(0), str(0)], torch.tensor(pc).float()


        elif not self.use_custom_batch:
            label = self.map_batch_labels([self.pc_cluster_labels[index]])
            pc = self.pc_cluster[index]
            pc = self.pc_norm(pc)
            return torch.tensor(label).long(), [0], torch.tensor(pc).float()
        
        else:
            batch_label = self.map_batch_labels([self.pc_cluster_labels[i] for i in self.custom_batches[index]])
            batch_pc = np.array([self.pc_cluster[i] for i in self.custom_batches[index]])
            batch_pc = np.array(list(map(self.pc_norm, batch_pc)))

            shuffle_batch = np.random.choice(len(batch_label), len(batch_label), replace=False)
            batch_label = batch_label[shuffle_batch]
            batch_pc = batch_pc[shuffle_batch]

            return torch.from_numpy(batch_label.astype(np.float32)), [0], torch.from_numpy(batch_pc.astype(np.float32))
    def __len__(self):
        if self.subset == 'test':
            return len(self.pc_cluster)
        elif not self.use_custom_batch:
            return len(self.pc_cluster)
        else:
            return len(self.custom_batches)
        
#%%
# config = cfg_from_yaml_file('./TractoGPT/cfgs/dataset_configs/qbx_goback_3L_hybrid.yaml')
# dataset = qbx_goback_3L_hybrid(config)
#%%
# cnt = 0
# for i in range(len(dataset.cluster)):
#     if(len(dataset.cluster[i]))<10:
#         cnt+=1
# print(cnt)
# print(len(np.unique(dataset.pc_cluster_labels, return_counts=True)[1]))
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
# # with open('./data/dataTrInF/qbx/val/sub-1254.pkl', 'rb') as f:
# #     data = pickle.load(f)
# # z = np.array(data[0])[:,3]
# # #%%
# # a,b = np.unique(z, return_counts=True)
# # max(b)
# # #%%
# # dataset[10]
# #%%
# print(dataset.custom_batches.shape)
#%%
# tot = 0
# for i in dataset.streamlines.keys():
#     tot+=len(dataset.streamlines[i])
# print(tot)
# tot = 0
# for i in dataset.clusters:
#     tot+=len(i)
# print(tot)