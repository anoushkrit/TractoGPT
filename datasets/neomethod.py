# %%
import sys
import os,json
import datetime
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.interpolate import CubicSpline
from multiprocessing import Pool, cpu_count, set_start_method
from dipy.tracking.streamline import select_random_set_of_streamlines, set_number_of_points
sys.path.append('/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/')
from utils.config import *
import gc, pickle
import copy
from .build import DATASETS

#%%
@DATASETS.register_module()
class neomethod(Dataset):

    def __init__(self,config):
        self.config = config
        self.data_path = config.data_path
        self.subset = config.subset
        self.custom_batch = config.custom_batch
        self.pts_per_sample = config.pts_per_sample
        self.samples_per_class_per_batch = config.samples_per_class_per_batch
        self.use_resampled_streamline = config.use_resampled_streamline
        self.label_mapping_path = config.label_mapping_path

        if self.use_resampled_streamline:
            raise NotImplementedError('Resampled Streamlines not implemented yet')

        self.label_int_mapping = self.load_label_mapping()
        self.create_custom_batches()
        self.load_complete_data()
    
    def load_label_mapping(self):
        with open(self.label_mapping_path, 'r') as f:
            label_mapping = json.load(f)
        
        return label_mapping

    def create_custom_batches(self):
        unique_labels, label_counts = np.unique(self.cluster_labels, return_counts=True)
        batch_size = self.samples_per_class_per_batch*len(unique_labels)
        num_batches = int(len(unique_labels)*min(label_counts)//self.samples_per_class_per_batch)
        self.custom_batches = np.zeros((num_batches, batch_size))

        for i, label in enumerate(unique_labels):
            label_indices = np.where(self.cluster_labels == label)[0]
            label_indices = np.random.choice(label_indices, self.samples_per_class_per_batch*num_batches, replace=False)
            label_indices = label_indices.reshape((num_batches, self.samples_per_class_per_batch))
            self.custom_batches[:, i*self.samples_per_class_per_batch:(i+1)*self.samples_per_class_per_batch] = label_indices

    def load_data(self, pth):
        with open(pth, 'rb') as f:
            data = pickle.load(f)
        
        return data
    def load_complete_data(self):

        subject_ids = os.listdir(self.data_path)
        inp = [os.path.join(self.data_path, i) for i in subject_ids]
        self.data = {}
        with Pool() as p:
            data = p.map(self.load_data, inp)
        
        for i in range(len(subject_ids)):
            self.data[subject_ids[i].stem] = data[i]

        del data
        gc.collect()
    
    def create_clusters_subject(self, args):
        data, subject_id = args
        cluster_ids = data['cluster_ids']
        unique_cluster_ids = np.unique(cluster_ids)
        clusters = []
        cluster_labels = []
        for i in unique_cluster_ids:
            streamline_idx = np.where(cluster_ids == i)[0]
            u_labels, u_counts= np.unique(data['labels'][streamline_idx], return_counts=True)
            label = u_labels[np.argmax(u_counts)]
            streamline_idx = [f'{subject_id}_{j}' for j in streamline_idx]
            clusters.append(streamline_idx)
            cluster_labels.append(label)
        
        return clusters, cluster_labels
    def create_clusters(self):

        subject_ids = list(self.data.keys())
        inp = [(self.data[i.stem], i.stem) for i in subject_ids]
        with Pool() as p:
            clusters = p.map(self.create_clusters_subject, subject_ids)

        self.clusters = []
        self.cluster_labels = []
        for i in len(clusters):
            self.clusters.extend(clusters[i][0])
            self.cluster_labels.extend(clusters[i][1])

        del clusters
        gc.collect()

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    def concat_streamlines(self,indices):
        club_streamlines = np.array([])  
        subject_id = indices[0].split('_')[0]      
        indices = [int(i.split('_')[1]) for i in indices]
        for idx in indices:
            if self.use_resampled_streamline:
                club_streamlines = np.concatenate((club_streamlines, self.resampled_streamlines[idx]))
            else:
                club_streamlines = np.concatenate((club_streamlines, self.data[subject_id]['streamlines'][idx]))     
        
        club_streamlines = self.pc_norm(club_streamlines)
        original_club_streamlines = copy.deepcopy(club_streamlines)
        while len(club_streamlines)+len(original_club_streamlines) < self.pts_per_sample:
            club_streamlines = np.concatenate((club_streamlines,original_club_streamlines))

        if len(original_club_streamlines) > self.pts_per_sample:
            club_streamlines = original_club_streamlines[np.random.choice(len(original_club_streamlines), self.pts_per_sample, replace=False)]
        else:
            club_streamlines = np.concatenate((club_streamlines,original_club_streamlines[np.random.choice(len(original_club_streamlines), self.pts_per_sample-len(club_streamlines), replace=False)]))
        
        return np.array(club_streamlines)

    def map_batch_labels(self, labels):
        int_labels = [int(self.label_int_mapping[class_label]) for class_label in labels]
        return np.array(int_labels)
    
    def __getitem__(self, index):
        if not self.custom_batch:
            label = self.map_batch_labels([self.cluster_labels[index]])
            pc = self.concat_streamlines(self.clusters[index])
            return torch.tensor(label).long(), [0], torch.tensor(pc).float()
        else:
            batch_label = self.map_batch_labels([self.cluster_labels[i] for i in self.custom_batches[index]])
            batch_pc = np.array([self.concat_streamlines(self.clusters[i]) for i in self.custom_batches[index]])

            shuffle_batch = np.random.choice(len(batch_label), len(batch_label), replace=False)
            batch_label = batch_label[shuffle_batch]
            batch_pc = batch_pc[shuffle_batch]

            return torch.tensor(batch_label).long(), [0], torch.tensor(batch_pc).float()
    def __len__(self):
        if not self.custom_batch:
            return len(self.clusters)
        else:
            return len(self.custom_batches)
        
#%%
# config = cfg_from_yaml_file('/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/cfgs/dataset_config/neomethod.yaml')
# dataset = neomethod(config)