# %%
import sys
import os,json
import datetime
import numpy as np
from pathlib import Path
from hashlib import sha256
from blist import blist
import torch
from dipy.segment.fss import FastStreamlineSearch
from dipy.io.streamline import load_tractogram
from torch.utils.data import Dataset
from scipy.interpolate import CubicSpline
from multiprocessing import Pool, cpu_count, set_start_method
from dipy.tracking.streamline import select_random_set_of_streamlines, set_number_of_points
from dipy.tracking.distances import approx_polygon_track as apt
sys.path.append('/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/')
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
    return blist(resampled_streamline)

def interp_streamline(args):
    streamline, pts = args
    old_points = np.linspace(0, 1, streamline.shape[0])
    new_points = np.linspace(0, 1, pts)

    # x_old = np.linspace(streamline[:, 0].min(), streamline[:, 0].max(), streamline.shape[0])
    # y_old = np.linspace(streamline[:, 1].min(), streamline[:, 1].max(), streamline.shape[0])
    # z_old = np.linspace(streamline[:, 2].min(), streamline[:, 2].max(), streamline.shape[0])

    # x_new = np.linspace(streamline[:, 0].min(), streamline[:, 0].max(), pts)
    # y_new = np.linspace(streamline[:, 1].min(), streamline[:, 1].max(), pts)
    # z_new = np.linspace(streamline[:, 2].min(), streamline[:, 2].max(), pts)
    
    x_interp = CubicSpline(old_points, streamline[:, 0], bc_type='natural')
    y_interp = CubicSpline(old_points, streamline[:, 1], bc_type='natural')
    z_interp = CubicSpline(old_points, streamline[:, 2], bc_type='natural')
    
    resampled_streamline = np.vstack([
        x_interp(new_points),
        y_interp(new_points),
        z_interp(new_points)
    ]).T
    
    return blist(resampled_streamline)
# def interp_streamline(args):
#     streamline, pts = args
    
#     # Compute cumulative distances along the streamline (parametric variable)
#     distances = np.sqrt(np.sum(np.diff(streamline, axis=0)**2, axis=1))
#     param = np.hstack([[0], np.cumsum(distances)])

#     # Normalize the parametric variable to the range [0, 1]
#     param /= param[-1]

#     # Interpolate along the parametric variable
#     new_param = np.linspace(0, 1, pts)
    
#     # Perform cubic spline interpolation
#     x_interp = CubicSpline(param, streamline[:, 0], bc_type='natural')
#     y_interp = CubicSpline(param, streamline[:, 1], bc_type='natural')
#     z_interp = CubicSpline(param, streamline[:, 2], bc_type='natural')
    
#     # Resample the streamline at the new parametric values
#     resampled_streamline = np.vstack([
#         x_interp(new_param),
#         y_interp(new_param),
#         z_interp(new_param)
#     ]).T
    
#     return resampled_streamline

def choose_upORdownsample(args):
    streamline, pts = args
    if pts > len(streamline):
        return interp_streamline((streamline, pts))
    else:
        return downsample((streamline, pts))

@DATASETS.register_module()
class single_streamline(Dataset):
    def __init__(self, config):
        self.config = config
        self.path = config.path
        self.subset = config.subset
        self.points_per_streamline = config.points_per_streamline
        self.samples_per_class = config.samples_per_class
        self.sample_per_subject = config.sample_per_subject
        self.num_samples_per_class_per_batch = config.num_samples_per_class_per_batch
        self.pts_per_sample = config.pts_per_sample
        self.label_mapping_path = config.label_mapping_path
        self.use_custom_batch = config.custom_batch
        self.min_batches = config.min_batches
        if self.subset == 'test':
            self.use_custom_batch = False

        print('Processing data for :',self.subset)
        self.batch_root_path = f'data/single_streamline/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/batches{self.points_per_streamline}'
        if not os.path.exists(self.batch_root_path):
            os.makedirs(self.batch_root_path)
        
        if not os.path.exists(self.label_mapping_path):
            raise FileNotFoundError("either the path of mapping json is wrong or it does not exist")
        else:
            with open(self.label_mapping_path,'r') as f:
                self.label_int_mapping = json.load(f)

        self.root_path = 'data/single_streamline'

        path = f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/streamlines.pkl'
        if not os.path.exists(path) or self.subset == 'test':
            self.load_streamlines_mp()
        else:
            self.streamlines = self.loadORprocess(path, self.load_streamlines_mp, title='streamlines')
            path = f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/subjectIds.pkl'
            self.subject_ids = self.loadORprocess(path, self.load_streamlines_mp, title='subjectIds')
            path = f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/labels.pkl'
            self.labels = self.loadORprocess(path, self.load_streamlines_mp, title='labels')
        path = f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/resampledStreamlines_{self.points_per_streamline}.pkl'
        self.resampled_streamlines = self.loadORprocess(path, self.fix_pts_in_streamlines, title='resampledStreamlines')
        print('batching start : ', datetime.datetime.now())
        if self.use_custom_batch:
            self.create_custom_batch()
        self.itr_count = 0

        print('end : ', datetime.datetime.now())

    def hash_streamline(self, streamline):
        return sha256(streamline.tobytes()).hexdigest()
    
    def loadORprocess(self, path, process_func, title):
        
        if not os.path.exists(path) or self.subset == 'test':
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
    
    def create_custom_batch(self):
        # stratified sampler
        unique_labels, label_counts = np.unique(self.labels, return_counts=True)
        print('min_label: ',min(label_counts))
        batch_size = self.num_samples_per_class_per_batch*len(unique_labels)
        num_batches = max(int(min(label_counts)//self.samples_per_class_per_batch), self.min_batches)
        self.custom_batches = np.zeros((num_batches, batch_size), dtype=int)

        for i, label in enumerate(unique_labels):
            label_indices = np.where(self.labels == label)[0]
            num_samples = self.num_samples_per_class_per_batch*num_batches
            full_rounds = num_samples//len(label_indices)
            remainder = num_samples%len(label_indices)
            new_label_indices = np.tile(label_indices, full_rounds)
            if remainder > 0:
                new_label_indices = np.concatenate((new_label_indices, np.random.choice(label_indices, remainder, replace=False)))
            np.random.shuffle(label_indices)

            label_indices = new_label_indices.reshape((num_batches, self.num_samples_per_class_per_batch))
            self.custom_batches[:, i*self.num_samples_per_class_per_batch:(i+1)*self.num_samples_per_class_per_batch] = label_indices
    
    def load_subject(self,subject_data):
        """ Helper function to process streamlines for a single subject. """
        subject, data_path, sample_per_subject, samples_per_class = subject_data
        subject_id = subject.name
        streamlines = []
        streamline_to_class = {}

        class_labels = [clss for clss in Path(os.path.join(data_path, subject, 'tracts')).iterdir()]
        class_labels = [clss for clss in class_labels if clss.suffix == '.trk' and clss.stem != 'CC_0' and clss.stem != 'CC']
        class_labels = [clss for clss in class_labels if clss.stem in self.label_int_mapping.keys()]
        for clss in class_labels:
            class_name = clss.stem

            cls_path = os.path.join(data_path, subject, 'tracts', clss)
            loaded_streamlines = load_stream(cls_path)
            if len(loaded_streamlines) > samples_per_class:
                loaded_streamlines = [loaded_streamlines[i] for i in np.random.choice(len(loaded_streamlines), samples_per_class, replace=False)]
            
            streamlines.extend(loaded_streamlines)
            streamline_to_class.update({self.hash_streamline(streamline): class_name for streamline in loaded_streamlines})

        if len(streamlines) > sample_per_subject:
            streamlines = [streamlines[i] for i in np.random.choice(len(streamlines), sample_per_subject, replace=False)]

        labels = [streamline_to_class[self.hash_streamline(streamline)] for streamline in streamlines]
        
        return subject_id, streamlines, labels

    def load_streamlines_mp(self):
        data_path = os.path.join(self.path, self.subset)
        subjects_classes = {}
        streamlines = []
        subject_ids = []
        labels = []

        subjects = [subject for subject in Path(data_path).iterdir() if os.path.isdir(subject) and subject.name != 'meta']
        if self.subset=='test' and type(self.config.subject_ids) == list: # using only the given subjects
            subjects = [subject for subject in subjects if subject.name in self.config.subject_ids]
        
        subject_data = [
            (subject, data_path, self.sample_per_subject, self.samples_per_class)
            for subject in subjects
        ]

        # Using multiprocessing to parallelize the loading
        with Pool(processes=os.cpu_count()) as pool:
            results = pool.map(self.load_subject, subject_data)

        for subject_id, subject_streamlines, subject_labels in results:
            subjects_classes[subject_id] = dict(zip(subject_labels, subject_streamlines))
            streamlines.extend(subject_streamlines)
            subject_ids.extend([int(subject_id)] * len(subject_streamlines))
            labels.extend(subject_labels)

        self.streamlines = blist(streamlines)
        self.subject_ids = np.array(subject_ids)
        self.labels = np.array(labels)
        
        if self.subset != 'test':
            with open(f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/streamlines.pkl', 'wb') as f:
                pickle.dump(self.streamlines, f)
            with open(f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/labels.pkl', 'wb') as f:
                pickle.dump(self.labels,f)
            with open(f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/subjectIds.pkl', 'wb') as f:
                pickle.dump(self.subject_ids, f)

        del streamlines
        del subject_ids
        del labels
        gc.collect()

    def fix_pts_in_streamlines(self):
        pts = self.points_per_streamline
        x = [(streamline, pts) for streamline in self.streamlines]
        with Pool(processes=os.cpu_count()) as pool:
            resampled_streamlines = pool.map(choose_upORdownsample, x)
        resampled_streamlines = blist(resampled_streamlines)

        if self.subset != 'test':
            with open(f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/resampledStreamlines_{self.points_per_streamline}.pkl', 'wb') as f:
                pickle.dump(resampled_streamlines, f)
        return resampled_streamlines
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def map_batch_labels(self, labels):
        int_labels = [int(self.label_int_mapping[class_label]) for class_label in labels]
        return np.array(int_labels)
    def __getitem__(self,index):
        if not self.use_custom_batch:
            return torch.from_numpy(np.array(self.label_int_mapping[self.labels[index]])), [str(self.subject_ids[index]), str(self.streamlines[index])], torch.from_numpy(self.resampled_streamlines[index].astype(np.float32))
            # return torch.from_numpy(self.label_int_mapping[self.labels[index]]), [0], torch.from_numpy(self.resampled_streamlines[index].astype(np.float32))
        else:
            streamline_idx = self.custom_batches[index]
            labels = self.labels[streamline_idx]
            labels = self.map_batch_labels(labels)
            batch_streamlines = np.array([self.pc_norm(self.resampled_streamlines[i]) for i in streamline_idx], dtype=np.float32)
            shuffle = np.random.choice(len(batch_streamlines), len(batch_streamlines), replace=False)

            labels = labels[shuffle]
            batch_streamlines = batch_streamlines[shuffle]

            return torch.from_numpy(labels), [0], torch.from_numpy(batch_streamlines)

    def __len__(self):
        if self.use_custom_batch: 
            return len(self.custom_batches)
        return len(self.streamlines)
    
#%%
# config = cfg_from_yaml_file('/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/cfgs/dataset_configs/single_streamline.yaml')
# x = single_streamline(config)
# #%%
# a,b,c = x[0]
# # %%
# import plotly.graph_objects as go
# def plot(streamlines, title):
#     fig = go.Figure()

#     # Loop through each streamline and plot
#     # for streamline in streamlines:
#     #     streamline = np.array(streamline)
#     fig.add_trace(go.Scatter3d(
#         x=streamlines[:, 0],  # x-coordinates
#         y=streamlines[:, 1],  # y-coordinates
#         z=streamlines[:, 2],  # z-coordinates
#         mode='lines',  
#             # Plotting lines for the streamlines
#         marker=dict(
#                     size=2,          # Size of the points
#                     opacity=0.8      # Adjust opacity for better visualization
#                 ), # Customize the line width if needed
#         name='Streamline'
#     ))

#     # Customize layout (optional)
#     fig.update_layout(
#         scene=dict(
#             xaxis_title='X',
#             yaxis_title='Y',
#             zaxis_title='Z'
#         ),
#         title=title
#     )

#     # Show the plot
#     fig.show()
#     i = np.random.choice(6,1)[0]
#     # fig.write_html(f'single_plots/{title}___{i}.html')
# # %%
# idx =  634
# plot(a[idx],c[idx])
# print('point in above plotted streamline :',len(a[idx]))