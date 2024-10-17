# %%
import sys
import numpy as np
from pathlib import Path
from blist import blist
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.interpolate import CubicSpline
from multiprocessing import Pool, cpu_count, set_start_method
# from dipy.tracking.streamline import select_random_set_of_streamlines, set_number_of_points
sys.path.append('/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/')
from utils.config import *
import copy
from memory_profiler import profile
from .build import DATASETS
import h5py

#%%
@DATASETS.register_module()
class FIESTA_testing(Dataset):

    def __init__(self,config):
        self.config = config
        self.hdf5_path = config.hdf5_path
        self.subset = config.subset
        self.pts_per_sample = config.pts_per_sample
        
        self.get_clusters_and_gt()
    
    def get_clusters_and_gt(self):
        self.clusters = []
        self.clusters_gt = []
        with h5py.File(self.hdf5_path, 'r') as f:
            for id in f[self.subset].keys():
                self.clusters.append(f[self.subset][id]['streamline'])
                self.clusters_gt.append(f[self.subset][id]['gt'])

    def pc_norm(pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx):
        pc = self.pc_norm(self.clusters[idx])
        label = self.clusters_gt[idx]
        pc_ = copy.deepcopy(pc)
        while len(pc)+len(pc_) < self.pts_per_sample:
            pc = np.concatenate((pc,pc_))
        if len(pc_) > self.pts_per_sample:
            pc = pc_[np.random.choice(len(pc_), self.pts_per_sample, replace=False)]
        else:
            pc = np.concatenate((pc,pc_[np.random.choice(len(pc_), self.pts_per_sample-len(pc), replace=False)]))
        
        return torch.from_numpy(np.array(label)), [], torch.from_numpy(pc.astype(np.float32))
    def __len__(self):
        return len(self.clusters)
