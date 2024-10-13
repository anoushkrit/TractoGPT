# %%
import sys
import os,json
import datetime
import numpy as np
from pathlib import Path
import torch
from hashlib import sha256
from tqdm import tqdm
from dipy.segment.fss import FastStreamlineSearch
from dipy.io.streamline import load_tractogram
from torch.utils.data import Dataset
from scipy.interpolate import CubicSpline
from multiprocessing import Pool
from dipy.tracking.streamline import select_random_set_of_streamlines, set_number_of_points
from dipy.tracking.distances import approx_polygon_track as apt
sys.path.append('/scratch/anoushkrit.scee.iitmandi/github/TractoGPT')
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
    
    return resampled_streamline

def choose_upORdownsample(args):
    streamline, pts = args
    if pts > len(streamline):
        return interp_streamline((streamline, pts))
    else:
        return downsample((streamline, pts))

@DATASETS.register_module()
class single_streamline_TrInF(Dataset):
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
        self.custom_batch = config.custom_batch 
        self.subject_ids = config.subject_ids

        if self.subset == 'testset':
            self.custom_batch = False

        print('Processing data/dataTrInF for :',self.subset)
        self.batch_root_path = f'data/dataTrInF/single_streamline/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/batches{self.points_per_streamline}'
        if not os.path.exists(self.batch_root_path):
            os.makedirs(self.batch_root_path)
        
        self.load_label_mapping()

        self.root_path = 'data/dataTrInF/single_streamline'

        path = f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/streamlines.pkl'
        if not os.path.exists(path) or self.subset=='testset': 
            self.load_streamlines_mp()
        else:
            self.streamlines = self.loadORprocess(path, self.load_streamlines_mp, title='streamlines')
            path = f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/subjectIds.pkl'
            self.subject_ids = self.loadORprocess(path, self.load_streamlines_mp, title='subjectIds')
            path = f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/labels.pkl'
            self.labels = self.loadORprocess(path, self.load_streamlines_mp, title='labels')
        path = f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/resampledStreamlines_{self.points_per_streamline}.pkl'
        self.resampled_streamlines = self.loadORprocess(path, self.fix_pts_in_streamlines, title='resampledStreamlines')

        if self.custom_batch:
            self.create_custom_batch()
        
        print('end : ', datetime.datetime.now())

    def load_label_mapping(self):
        if not os.path.exists(self.label_mapping_path):
            raise FileNotFoundError("either the path of mapping json is wrong or it does not exist")
        else:
            with open(self.label_mapping_path,'r') as f:
                self.label_int_mapping = json.load(f)
    def loadORprocess(self, path, process_func, title):
        
        if not os.path.exists(path):
            dirname = os.path.dirname(path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            print('Processing data/dataTrInF of :',title)
            data = process_func()
        else:
            print('Loading processed data/dataTrInF for :',title, path)
            with open(path, 'rb') as f:
                data = pickle.load(f)
        return data
    
    def create_custom_batch(self):
        # stratified sampler
        labels, counts = np.unique(self.labels, return_counts=True)
        self.num_batches = np.min(counts)//self.num_samples_per_class_per_batch
        self.batch_size = len(labels)*self.num_samples_per_class_per_batch
        # print(self.num_batches, self.batch_size)
        if not os.path.exists(f'{self.batch_root_path}/custom_batches_{self.batch_size}'):
            os.makedirs(f'{self.batch_root_path}/custom_batches_{self.batch_size}')
        
        x = np.zeros((self.batch_size, self.num_batches), dtype=np.int32)
        for i in range(len(labels)):
            indices = np.where(self.labels == labels[i])[0]
            z = np.random.choice(len(indices), self.num_batches*self.num_samples_per_class_per_batch, replace=False)
            for j in range(self.num_samples_per_class_per_batch):
                x[i*self.num_samples_per_class_per_batch+j] = indices[z][j*self.num_batches:(j+1)*self.num_batches]

        for i in range(self.num_batches):
            batch = x[:,i]
            respective_labels = self.labels[batch]
            with open(f'{self.batch_root_path}/custom_batches_{self.batch_size}/batch_{i}.pkl', 'wb') as f:
                pickle.dump((batch, respective_labels),f)
    
    def hash_streamline(self, streamline):
        return sha256(streamline.tobytes()).hexdigest()
    def load_subject(self,subject_data):
        """ Helper function to process streamlines for a single subject. """
        subject, data_path, sample_per_subject, samples_per_class = subject_data
        subject_id = subject.name
        streamlines = []
        streamline_to_class = {}

        dir_content = Path(os.path.join(data_path, subject, 'tractography')).iterdir()
        dir_content = sorted([p for p in dir_content if p.is_file() and p.suffix == '.trk'])

        for clss in dir_content:
            class_name = clss.stem.split('__')[-1]
            if class_name not in self.label_int_mapping.keys():
                continue

            cls_path = os.path.join(data_path, subject, 'tractography', clss)
            loaded_streamlines = load_stream(cls_path)

            # Optionally downsample if needed
            if len(loaded_streamlines) > samples_per_class:
                # loaded_streamlines = select_random_set_of_streamlines(loaded_streamlines, self.samples_per_class)
                loaded_streamlines = [loaded_streamlines[i] for i in np.random.choice(len(loaded_streamlines), samples_per_class, replace=False)]
            
            streamlines.extend(loaded_streamlines)
            streamline_to_class.update({self.hash_streamline(streamline): class_name for streamline in loaded_streamlines})

        if len(streamlines) > sample_per_subject:
            # streamlines = select_random_set_of_streamlines(streamlines, sample_per_subject)
            streamlines = [streamlines[i] for i in np.random.choice(len(streamlines), sample_per_subject, replace=False)]

        labels = [streamline_to_class[self.hash_streamline(streamline)] for streamline in streamlines]
        return subject_id, streamlines, labels

    def load_streamlines_mp(self):
        data_path = os.path.join(self.path, self.subset)
        streamlines = []
        subject_ids = []
        labels = []

        subjects = [subject for subject in Path(data_path).iterdir() if os.path.isdir(subject)]
        if self.subset=='testset' and type(self.config.subject_ids) == list: # using only the given subjects
            subjects = [subject for subject in subjects if subject.name in self.config.subject_ids]
        
        subject_data = [(subject, data_path, self.sample_per_subject, self.samples_per_class)for subject in subjects]

        # Using multiprocessing to parallelize the loading
        with Pool(processes=os.cpu_count()) as pool:
            results = pool.map(self.load_subject, subject_data)

        for subject_id, subject_streamlines, subject_labels in results:
            streamlines.extend(subject_streamlines)
            subject_ids.extend([subject_id] * len(subject_streamlines))
            labels.extend(subject_labels)

        print(len(labels))
        self.streamlines = streamlines
        self.subject_ids = np.array(subject_ids)
        self.labels = np.array(labels)
        
        if self.subset != 'testset':
            with open(f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/streamlines.pkl', 'wb') as f:
                pickle.dump(self.streamlines, f)
            with open(f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/labels.pkl', 'wb') as f:
                pickle.dump(self.labels,f)
            with open(f'{self.root_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/subjectIds.pkl', 'wb') as f:
                pickle.dump(self.subject_ids, f)

    def fix_pts_in_streamlines(self):
        pts = self.points_per_streamline
        x = [(streamline, pts) for streamline in self.streamlines]
        with Pool(processes=os.cpu_count()) as pool:
            resampled_streamlines = pool.map(choose_upORdownsample, x)
        resampled_streamlines = np.array(resampled_streamlines)

        if self.subset != 'testset':
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
        if not self.custom_batch:
            return torch.from_numpy(np.array(self.label_int_mapping[self.labels[index]])), [self.subject_ids[index], str(self.streamlines[index])], torch.from_numpy(np.array(self.pc_norm(self.resampled_streamlines[index]), dtype=np.float32))

        with open(f'{self.batch_root_path}/custom_batches_{self.batch_size}/batch_{index}.pkl', 'rb') as f:
            batch_streamlines_indices, batch_labels = pickle.load(f)
        batch_pc = np.zeros((self.batch_size, self.pts_per_sample, 3))

        for i,idx in enumerate(batch_streamlines_indices):
            batch_pc[i] = self.resampled_streamlines[idx].copy()
            batch_pc[i] = batch_pc[i][np.random.choice(len(batch_pc[i]), self.pts_per_sample, replace=False)]
            batch_pc[i] = self.pc_norm(batch_pc[i])
        batch_labels = np.array(batch_labels)
        mapped_labels = self.map_batch_labels(batch_labels)
        shuffle_batch = np.random.choice(len(batch_pc),len(batch_pc), replace=False)
        return torch.from_numpy(mapped_labels[shuffle_batch]), [0] ,torch.from_numpy(batch_pc[shuffle_batch].astype(np.float32)) 
        # return self.streamlines, self.resampled_streamlines, self.labels
    def __len__(self):
        if not self.custom_batch:
            return len(self.labels)
        return self.num_batches
    
#%%
# config = cfg_from_yaml_file('/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/cfgs/dataset_configs/single_streamline_TrInF.yaml')
# x = single_streamline_TrInF(config)
# #%%
# len(x.labels)
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