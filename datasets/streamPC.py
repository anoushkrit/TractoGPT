# %%
import sys
import os,json
import datetime
import numpy as np
from pathlib import Path
import torch
from sklearn.neighbors import NearestNeighbors
from dipy.segment.fss import FastStreamlineSearch
from dipy.io.streamline import load_tractogram
from torch.utils.data import Dataset
from scipy.interpolate import CubicSpline
from multiprocessing import Pool, cpu_count, set_start_method
from dipy.tracking.streamline import select_random_set_of_streamlines, set_number_of_points
from functools import partial
# from dipy.se
from dipy.tracking.distances import approx_polygon_track as apt
# sys.path.append('../')
sys.path.append('./TractoGPT/')
from utils.config import *
import gc, pickle
from dipy.tracking.streamline import select_random_set_of_streamlines
import copy
from .build import DATASETS


def load_stream(trk_path):
    bundle_track = load_tractogram(str(trk_path), 'same', bbox_valid_check=False)
    return list(bundle_track.streamlines)

def downsample(args):
    streamline, pts = args
    streamline = apt(streamline,.1)
    if len(streamline) != pts: 
        resampled_streamline = interp_streamline((streamline, pts))
    else:
        resampled_streamline = streamline
    return resampled_streamline

def interp_streamline(args):
    streamline, pts = args
    # old_points = np.linspace(0, 1, streamline.shape[0])
    # new_points = np.linspace(0, 1, pts)

    x_old = np.linspace(streamline[:, 0].min(), streamline[:, 0].max(), streamline.shape[0])
    y_old = np.linspace(streamline[:, 1].min(), streamline[:, 1].max(), streamline.shape[0])
    z_old = np.linspace(streamline[:, 2].min(), streamline[:, 2].max(), streamline.shape[0])

    x_new = np.linspace(streamline[:, 0].min(), streamline[:, 0].max(), pts)
    y_new = np.linspace(streamline[:, 1].min(), streamline[:, 1].max(), pts)
    z_new = np.linspace(streamline[:, 2].min(), streamline[:, 2].max(), pts)
    
    x_interp = CubicSpline(x_old, streamline[:, 0], bc_type='natural')
    y_interp = CubicSpline(y_old, streamline[:, 1], bc_type='natural')
    z_interp = CubicSpline(z_old, streamline[:, 2], bc_type='natural')
    
    resampled_streamline = np.vstack([
        x_interp(x_new),
        y_interp(y_new),
        z_interp(z_new)
    ]).T
    
    return resampled_streamline

def choose_upORdownsample(args):
    streamline, pts = args
    if pts > len(streamline):
        return interp_streamline((streamline, pts))
    else:
        return downsample((streamline, pts))

def FSS(args):
    streamlines, main_streamline, radius, indices = args
    fs_tree_af = FastStreamlineSearch(ref_streamlines=streamlines, max_radius=radius)
    coo_mdist_mtx = fs_tree_af.radius_search(main_streamline, radius=radius)
    ids_ref = np.unique(coo_mdist_mtx.col) 
    ids_ref = [indices[i] for i in ids_ref]
    return ids_ref

@DATASETS.register_module()
class streamPC(Dataset):
    def __init__(self, config):
        self.config = config
        self.path = config.path
        self.subset = config.subset
        self.points_per_streamline = config.points_per_streamline
        self.num_local_neighbours = config.num_local_neighbours
        self.FSSradius = config.FSSradius
        self.samples_per_class = config.samples_per_class
        self.sample_per_subject = config.sample_per_subject
        self.num_samples_per_class_per_batch = config.num_samples_per_class_per_batch
        self.pts_per_sample = config.pts_per_sample
        self.debug_mode = config.debug_mode
        self.omitted_streamlines = []
        self.label_mapping_path = config.label_mapping_path
        self.local_streamline_threshold_count = config.local_streamline_threshold_count
        self.use_resampled_streamline = config.use_resampled_streamline

        print('Processing data for :',self.subset)
        self.batch_root_path = f'data/s3FSSknn/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/batches_streamPC{self.FSSradius}{self.num_local_neighbours}{self.local_streamline_threshold_count}'
        if not os.path.exists(self.batch_root_path):
            os.makedirs(self.batch_root_path)
        
        if not os.path.exists(self.label_mapping_path):
            print("either the path of mapping json is wrong or \nit does not exist")
            raise FileNotFoundError()
        else:
            with open(self.label_mapping_path,'r') as f:
                self.label_int_mapping = json.load(f)

        root_path = 'data/s3FSSknn/'
        self.root_path = root_path
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        path = f'{root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/streamlines.pkl'
        if not os.path.exists(path):
            self.load_streamlines_mp()
        else:
            self.streamlines = self.loadORprocess(path, self.load_streamlines_mp, title='streamlines')
            path = f'{root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/subjectIds.pkl'
            self.subject_ids = self.loadORprocess(path, self.load_streamlines_mp, title='subjectIds')
            path = f'{root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/labels.pkl'
            self.labels = self.loadORprocess(path, self.load_streamlines_mp, title='labels')
        path = f'{root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/resampledStreamlines_{self.points_per_streamline}.pkl'
        self.resampled_streamlines = self.loadORprocess(path, self.fix_pts_in_streamlines, title='resampledStreamlines')
        path = f'{root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/FSSidx/FSSradius{self.FSSradius}.pkl'
        self.FSS_indices = self.loadORprocess(path, self.apply_FSS, title = 'FSSidx')
        path = f'{root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/FSSidx/ommitedIdx_FSSradius{self.FSSradius}/ommitedIdx_{self.local_streamline_threshold_count}.pkl'
        self.omitted_streamlines = self.loadORprocess(path, self.get_ommited_indices, title = 'ommitedIdx')
        path = f'{root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/chamferIdx/FSSradius{self.FSSradius}_numSL{self.num_local_neighbours}.pkl'
        self.chamfer_indices = self.loadORprocess(path, self.get_local_streamlines_subjectwise_CD, title = 'chamferIdx')
        
        print('batching start : ', datetime.datetime.now())
        print(len(self.omitted_streamlines))
        self.omitted_streamlines = np.concatenate((self.omitted_streamlines,np.where(self.labels=='CC')[0]))
        self.omitted_streamlines = np.concatenate((self.omitted_streamlines,np.where(self.labels=='CC_0')[0]))
        print(len(self.omitted_streamlines))
        self.create_custom_batch()
        self.itr_count = 0

        print('end : ', datetime.datetime.now())

    def loadORprocess(self, path, process_func, title):
        if not os.path.exists(path):
            dirname = os.path.dirname(path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            print('Processing data of :',title)
            data = process_func()
        else:
            print('Loading processed data for :',title, path)
            with open(path, 'rb') as f:
                data = pickle.load(f)
        return data
    def update_count_on_ommitted_streamlines(self, original_label, original_count):
        new_temp = self.labels[self.omitted_streamlines]
        ommited_labels, ommited_count = np.unique(new_temp[new_temp!='CC'], return_counts=True)

        for i in range(len(ommited_labels)):
            original_count[original_label == ommited_labels[i]] -= ommited_count[i]
        
        return original_count
    def create_custom_batch(self):
        # stratified sampler
        labels, counts = np.unique(self.labels[self.labels!='CC'], return_counts=True)
        updated_counts = self.update_count_on_ommitted_streamlines(labels, counts)
        self.num_batches = np.min(updated_counts)//self.num_samples_per_class_per_batch
        self.batch_size = len(labels)*self.num_samples_per_class_per_batch
        # print(self.num_batches, self.batch_size)
        if not os.path.exists(f'{self.batch_root_path}/custom_batches_{self.batch_size}'):
            os.makedirs(f'{self.batch_root_path}/custom_batches_{self.batch_size}')
        
        x = np.zeros((self.batch_size, self.num_batches), dtype=np.int32)
        for i in range(len(labels)):
            indices = np.where(self.labels == labels[i])[0]
            indices = indices[~np.isin(indices, self.omitted_streamlines)]
            # print(indices.shape)
            z = np.random.choice(len(indices), self.num_batches*self.num_samples_per_class_per_batch, replace=False)
            for j in range(self.num_samples_per_class_per_batch):
                x[i*self.num_samples_per_class_per_batch+j] = indices[z][j*self.num_batches:(j+1)*self.num_batches]

        for i in range(self.num_batches):
            batch = x[:,i]
            local_streamlines = []
            respective_labels = []
            for j in range(len(batch)):
                local_streamlines.append(list(self.chamfer_indices[batch[j]]))
                respective_labels.append((labels[j//self.num_samples_per_class_per_batch]))
            with open(f'{self.batch_root_path}/custom_batches_{self.batch_size}/batch_{i}.pkl', 'wb') as f:
                pickle.dump((batch, local_streamlines, respective_labels),f)
    
    def load_subject(self,subject_data):
        """ Helper function to process streamlines for a single subject. """
        subject, data_path, sample_per_subject, samples_per_class = subject_data
        subject_id = subject.name
        streamlines = []
        streamline_to_class = {}

        for clss in Path(os.path.join(subject, 'tracts')).iterdir():
            if clss.suffix != '.trk' or clss.stem == 'CC_0' or clss.stem == 'CC':
                continue
            if clss.stem not in self.label_int_mapping.keys():
                continue
            
            class_name = clss.stem
            cls_path = os.path.join(data_path, subject, 'tracts', clss)
            
            loaded_streamlines = load_stream(cls_path)
            # Optionally downsample if needed
            if len(loaded_streamlines) > samples_per_class:
                # loaded_streamlines = select_random_set_of_streamlines(loaded_streamlines, self.samples_per_class)
                loaded_streamlines = [loaded_streamlines[i] for i in np.random.choice(len(loaded_streamlines), samples_per_class, replace=False)]
            
            streamlines.extend(loaded_streamlines)
            streamline_to_class.update({str(streamline): class_name for streamline in loaded_streamlines})

        if len(streamlines) > sample_per_subject:
            # streamlines = select_random_set_of_streamlines(streamlines, sample_per_subject)
            streamlines = [streamlines[i] for i in np.random.choice(len(streamlines), sample_per_subject, replace=False)]

        labels = [streamline_to_class[str(streamline)] for streamline in streamlines]
        
        return subject_id, streamlines, labels

    def load_streamlines_mp(self):
        data_path = os.path.join(self.path, self.subset)
        subjects_classes = {}
        streamlines = []
        subject_ids = []
        labels = []

        subject_data = [
            (subject, data_path, self.sample_per_subject, self.samples_per_class)
            # for subject in Path(data_path).iterdir() if subject.name == '599671' or subject.name == '644044'
            for subject in Path(data_path).iterdir() if subject.name != 'meta' and os.path.isdir(subject)
        ]

        # Using multiprocessing to parallelize the loading
        with Pool(processes=os.cpu_count()) as pool:
            results = pool.map(self.load_subject, subject_data)

        for subject_id, subject_streamlines, subject_labels in results:
            subjects_classes[subject_id] = dict(zip(subject_labels, subject_streamlines))
            streamlines.extend(subject_streamlines)
            subject_ids.extend([int(subject_id)] * len(subject_streamlines))
            labels.extend(subject_labels)

        self.streamlines = streamlines
        self.subject_ids = np.array(subject_ids)
        self.labels = np.array(labels)
        
        with open(f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/streamlines.pkl', 'wb') as f:
            pickle.dump(self.streamlines, f)
        with open(f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/labels.pkl', 'wb') as f:
            pickle.dump(self.labels,f)
        with open(f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/subjectIds.pkl', 'wb') as f:
            pickle.dump(self.subject_ids, f)
        return subjects_classes

    def chamfer_distance(self,x,y):
        # x: Tensor of shape (batch_size, num_points_x, 3)
        # y: Tensor of shape (batch_size, num_points_y, 3)
        
        # Compute pairwise distance between points in x and y
        diff = x.unsqueeze(2) - y.unsqueeze(1)  # (batch_size, num_points_x, num_points_y, 3)
        dist = torch.sum(diff ** 2, dim=-1)  # (batch_size, num_points_x, num_points_y)
        
        # Compute the nearest distance from x to y and y to x
        min_dist_x_to_y, _ = torch.min(dist, dim=2)  # (batch_size, num_points_x)
        min_dist_y_to_x, _ = torch.min(dist, dim=1)  # (batch_size, num_points_y)
        
        # Sum of the Chamfer distances
        chamfer_dist = torch.mean(min_dist_x_to_y, dim=1) + torch.mean(min_dist_y_to_x, dim=1)  # (batch_size)
        
        return chamfer_dist.cpu()

    def get_local_streamlines_subjectwise_CD(self):
        """Find local streamlines using Chamfer distance (CUDA implementation)."""
        
        chamfer_indices = np.zeros((len(self.streamlines), self.num_local_neighbours), dtype=np.int32)
        
        for idx in range(len(self.resampled_streamlines)):

            main_streamline = torch.tensor(self.resampled_streamlines[idx], device='cuda')
            search_streamlines = torch.tensor(self.resampled_streamlines[self.FSS_indices[idx]], device='cuda')
            
            if len(self.FSS_indices[idx]) > self.num_local_neighbours:
                chamfer_distance = self.chamfer_distance(main_streamline.unsqueeze(0), search_streamlines)
                sorted_indices = torch.argsort(chamfer_distance).cpu().numpy()
                chamfer_indices[idx] = np.array([self.FSS_indices[idx][i] for i in sorted_indices[:self.num_local_neighbours]])
            else:
                extended_indices = (self.FSS_indices[idx] * ((self.num_local_neighbours // len(self.FSS_indices[idx])) + 1))[:self.num_local_neighbours]
                chamfer_indices[idx] = np.array(extended_indices)
        
        # Save indices to file
        with open(f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/chamferIdx/FSSradius{self.FSSradius}_numSL{self.num_local_neighbours}', 'wb') as f:
            pickle.dump(chamfer_indices, f)
        return chamfer_indices   
    
    def fix_pts_in_streamlines(self):
        pts = self.points_per_streamline
        x = [(streamline, pts) for streamline in self.streamlines]
        with Pool(processes=os.cpu_count()) as pool:
            resampled_streamlines = pool.map(choose_upORdownsample, x)
        resampled_streamlines = np.array(resampled_streamlines)

        with open(f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/resampledStreamlines_{self.points_per_streamline}.pkl', 'wb') as f:
            pickle.dump(resampled_streamlines, f)
        return resampled_streamlines
    
    def apply_FSS(self):
        FSS_indices = [0]*len(self.streamlines)
        for subject_id in np.unique(self.subject_ids):
            batch_indices = np.where(self.subject_ids == subject_id)[0]
            inp = [([self.streamlines[i] for i in batch_indices],
                    [self.streamlines[batch_indices[i]]],
                    self.FSSradius, batch_indices) for i in range(len(batch_indices))]
            with Pool() as pool:
                FSS_subject_indices = pool.map(FSS, inp)
            for i in range(len(batch_indices)):
                FSS_indices[batch_indices[i]] = FSS_subject_indices[i]

        with open(f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/FSSidx/FSSradius{self.FSSradius}.pkl', 'wb') as f:    
            pickle.dump(FSS_indices, f)

    def get_ommited_indices(self):
        ommited_indices = []
        for i in range(len(self.FSS_indices)):
            if len(self.FSS_indices[i]) <= self.local_streamline_threshold_count:
                ommited_indices.append(i)

        ommited_indices = np.array(ommited_indices)
        with open(f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/FSSidx/ommitedIdx_FSSradius{self.FSSradius}/ommitedIdx_{self.local_streamline_threshold_count}.pkl', 'wb') as f:    
            pickle.dump(ommited_indices, f)
        return ommited_indices
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    
    def concat_streamlines(self,indices):
        if self.use_resampled_streamline:
            club_streamlines = self.resampled_streamlines[indices[0]]
        else:
            club_streamlines = self.streamlines[indices[0]]
        for i in range(1,len(indices)):
            if self.use_resampled_streamline:
                club_streamlines = np.concatenate((club_streamlines, self.resampled_streamlines[indices[i]]))
            else:
                club_streamlines = np.concatenate((club_streamlines, self.streamlines[indices[i]]))     
        
        original_club_streamlines = np.array(club_streamlines)
        club_streamlines = original_club_streamlines

        while len(club_streamlines)+len(original_club_streamlines) < self.pts_per_sample:
            club_streamlines = np.concatenate((club_streamlines,original_club_streamlines))

        if len(original_club_streamlines) > self.pts_per_sample:
            club_streamlines = original_club_streamlines[np.random.choice(len(original_club_streamlines), self.pts_per_sample, replace=False)]
        else:
            # print(np.shape(np.random.choice(original_club_streamlines, self.pts_per_sample-len(club_streamlines), replace=False)))
            club_streamlines = np.concatenate((club_streamlines,original_club_streamlines[np.random.choice(len(original_club_streamlines), self.pts_per_sample-len(club_streamlines), replace=False)]))
        
        return np.array(club_streamlines)
    def map_batch_labels(self, labels):
        int_labels = [int(self.label_int_mapping[class_label]) for class_label in labels]
        return np.array(int_labels)
    def __getitem__(self,index):
        self.itr_count+=1
        with open(f'{self.batch_root_path}/custom_batches_{self.batch_size}/batch_{index}.pkl', 'rb') as f:
            main_streamline_indices, batch_streamlines_indices, batch_labels = pickle.load(f)
        
        batch_pc = np.zeros((self.batch_size, self.pts_per_sample, 3))
        for j in range(self.batch_size):
            club_streamlines = self.concat_streamlines(batch_streamlines_indices[j])
            batch_pc[j] = self.pc_norm(club_streamlines)
        main_streamlines = self.resampled_streamlines[main_streamline_indices]
        main_streamlines = np.array(main_streamlines, dtype=np.float32)
        for i in range(self.batch_size):
            main_streamlines[i] = self.pc_norm(main_streamlines[i])
        batch_labels = np.array(batch_labels)
        mapped_labels = self.map_batch_labels(batch_labels)
        if self.debug_mode:
            return batch_labels, batch_pc, self.chamfer_indices, self.FSS_indices, self.labels, self.streamlines, self.subject_ids
        else:
            shuffle_batch = np.random.choice(len(batch_pc),len(batch_pc), replace=False)
            return torch.from_numpy(mapped_labels[shuffle_batch]), torch.from_numpy(main_streamlines[shuffle_batch]), torch.from_numpy(batch_pc[shuffle_batch].astype(np.float32))
    def __len__(self):
        return self.num_batches
    
#%%
# config = cfg_from_yaml_file('./TractoGPT/cfgs/dataset_configs/s3FSSknn.yaml')
# x = s3FSSknn(config)
#%%
# a,b,_,_,_,_,_ = x[0]
# %%
# import plotly.graph_objects as go
# def plot(mai,streamlines):
#     fig = go.Figure()

#     # Loop through each streamline and plot
#     # for streamline in streamlines:
#     #     streamline = np.array(streamline)
#     fig.add_trace(go.Scatter3d(
#         x=streamlines[:, 0],  # x-coordinates
#         y=streamlines[:, 1],  # y-coordinates
#         z=streamlines[:, 2],  # z-coordinates
#         mode='markers',  
#             # Plotting lines for the streamlines
#         marker=dict(
#                     size=2,          # Size of the points
#                     opacity=0.8      # Adjust opacity for better visualization
#                 ), # Customize the line width if needed
#         name='Streamline'
#     ))
#     fig.add_trace(go.Scatter3d(
#         x=mai[:, 0],  # x-coordinates
#         y=mai[:, 1],  # y-coordinates
#         z=mai[:, 2],  # z-coordinates
#         mode='markers',  
#             # Plotting lines for the streamlines
#         marker=dict(
#                     color = 'red',
#                     size=2,          # Size of the points
#                     opacity=0.8      # Adjust opacity for better visualization
#                 ), # Customize the line width if needed
#         name='main'
#     ))

#     # Customize layout (optional)
#     fig.update_layout(
#         scene=dict(
#             xaxis_title='X',
#             yaxis_title='Y',
#             zaxis_title='Z'
#         ),
#         title='3D Streamlines'
#     )

#     # Show the plot
#     fig.show()
# # %%
# a,b,c = x[2]
# # %%
# s = 110
# plot(c[0][s],c[1][s])
