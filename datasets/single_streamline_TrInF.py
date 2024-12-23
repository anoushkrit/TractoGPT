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
sys.path.append('./TractoGPT')
from utils.config import *
from utils.logger import *
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
        self.use_custom_batch = config.use_custom_batch 
        self.subject_ids = config.subject_ids

        if self.subset == 'testset' or self.subset == 'validset':
            self.use_custom_batch = False

        print_log(f'[DATASETS] Processing data/dataTrInF for : {self.subset}', logger='single_streamline_TrInF')
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

        if self.use_custom_batch:
            self.create_custom_batch()
        
        print_log(f'[DATASETS] end : {datetime.datetime.now()}', logger='single_streamline_TrInF')

        self.itr_count = 0
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
            print_log(f'[DATASETS] Processing data/dataTrInF of : {title}', logger='single_streamline_TrInF')
            data = process_func()
        else:
            print_log(f'[DATASETS] Loading processed data/dataTrInF for : {title}', logger='single_streamline_TrInF')
            with open(path, 'rb') as f:
                data = pickle.load(f)
        return data
    
    def create_custom_batch(self):
        # stratified sampler
        unique_labels, label_counts = np.unique(self.labels, return_counts=True)

        print_log(f'[DATASETS] minimum label count in complete data loaded for {self.subset}: {min(label_counts)}', logger='single_streamline_TrInF')
        batch_size = self.num_samples_per_class_per_batch*len(unique_labels)
        num_batches = max(int(min(label_counts)//self.num_samples_per_class_per_batch), self.config.min_batches)
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
        if self.subset == 'testset':
            return torch.from_numpy(np.array(self.label_int_mapping[self.labels[index]])), [self.subject_ids[index], str(self.streamlines[index])], torch.from_numpy(np.array(self.pc_norm(self.resampled_streamlines[index]), dtype=np.float32))
        elif not self.use_custom_batch:
            return torch.from_numpy(np.array(self.label_int_mapping[self.labels[index]])), [0], torch.from_numpy(np.array(self.pc_norm(self.resampled_streamlines[index]), dtype=np.float32))

        else:
            batch_streamline_idx = self.custom_batches[index]
            batch_labels = self.map_batch_labels(self.labels[batch_streamline_idx])
            batch_streamlines = self.resampled_streamlines[batch_streamline_idx]
            batch_streamlines = np.array(list(map(self.pc_norm, batch_streamlines)), dtype=np.float32)

            shuffle = np.random.choice(batch_streamlines.shape[0], batch_streamlines.shape[0], replace=False)

            batch_labels = batch_labels[shuffle]
            batch_streamlines = batch_streamlines[shuffle]

            return torch.from_numpy(batch_labels), [0], torch.from_numpy(batch_streamlines)
    def __len__(self):
        if not self.use_custom_batch:
            return len(self.labels)
        return len(self.custom_batches)
    
#%%
# config = cfg_from_yaml_file('./TractoGPT/cfgs/dataset_configs/single_streamline_TrInF.yaml')
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
# print_log('[DATASETS] point in above plotted streamline :',len(a[idx]))