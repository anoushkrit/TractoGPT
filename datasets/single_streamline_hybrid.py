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
from blist import blist


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

@DATASETS.register_module()
class single_streamline_hybrid(Dataset):
    def __init__(self, config):
        self.config = config
        self.path1 = config.path1
        self.path2 = config.path2
        self.save_path = config.save_path
        self.subset = config.subset
        self.points_per_streamline = config.points_per_streamline
        self.samples_per_class = config.samples_per_class
        self.sample_per_subject = config.sample_per_subject
        self.num_samples_per_class_per_batch = config.num_samples_per_class_per_batch
        self.pts_per_sample = config.pts_per_sample
        self.label_mapping_path = config.label_mapping_path
        self.load_label_mapping()
        self.use_custom_batch = config.use_custom_batch 
        self.subject_ids = config.subject_ids

        if self.subset == 'test' or self.subset == 'val':
            self.use_custom_batch = False

        if 'TractoInferno' in self.path1:
            if self.subset == 'train':
                self.path1 = os.path.join(self.path1, 'trainset')
            elif self.subset == 'val':
                self.path1 = os.path.join(self.path1, 'validset')
            elif self.subset == 'test':
                self.path1 = os.path.join(self.path1, 'testset')
            self.path2 = os.path.join(self.path2, self.subset)
        else:
            if self.subset == 'train':
                self.path2 = os.path.join(self.path2, 'trainset')
            elif self.subset == 'val':
                self.path2 = os.path.join(self.path2, 'validset')
            elif self.subset == 'test':
                self.path2 = os.path.join(self.path2, 'testset')
            self.path1 = os.path.join(self.path1, self.subset)

        path = f'{self.save_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/streamlines.pkl'
        if not os.path.exists(path) or self.subset=='test': 
            self.load_streamlines_mp([self.path1, self.path2])
        else:
            self.streamlines = self.loadORprocess(path, self.load_streamlines_mp, title='streamlines')
            path = f'{self.save_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/subjectIds.pkl'
            self.subject_ids = self.loadORprocess(path, self.load_streamlines_mp, title='subjectIds')
            path = f'{self.save_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/labels.pkl'
            self.labels = self.loadORprocess(path, self.load_streamlines_mp, title='labels')
        path = f'{self.save_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/resampledStreamlines_{self.points_per_streamline}.pkl'
        if self.config.use_resampled_streamlines:
            self.resampled_streamlines = self.loadORprocess(path, self.fix_pts_in_streamlines, title='resampledStreamlines')

        if self.use_custom_batch:
            self.create_custom_batch()
        
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
        unique_labels, label_counts = np.unique(self.labels, return_counts=True)
        print(len(unique_labels))
        print(label_counts)
        print(f'minimum label count in complete data loaded for {self.subset}:',min(label_counts))
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

        if "HCP" in data_path:
            dataset = '105HCP'
        elif "TractoInferno" in data_path:
            dataset = 'TractoInferno'
        else:
            raise FileNotFoundError('No tractography or tracts folder found in subject directory')
        
        # if dataset == 'TractoInferno':
        #     subset = 'trainset' if self.subset == 'train' else subset
        #     subset = 'validset' if self.subset == 'val' else subset
        #     subset = 'testset' if self.subset == 'test' else subset
        
        if dataset == 'TractoInferno':
            dir_content = Path(os.path.join(data_path, subject, 'tractography')).iterdir()
        elif dataset == '105HCP':
            dir_content = Path(os.path.join(data_path, subject, 'tracts')).iterdir()
        else:
            raise FileNotFoundError('No tractography or tracts folder found in subject directory')
        # dir_content = Path(os.path.join(data_path, subject)).iterdir()
        dir_content = sorted([p for p in dir_content if p.is_file() and p.suffix == '.trk'])
        for clss in dir_content:

            if dataset == 'TractoInferno':
                class_name = clss.stem.split('__')[-1]
            elif dataset == '105HCP':
                class_name = clss.stem
            else:
                raise FileNotFoundError('No tractography or tracts folder found in subject directory')

            if class_name not in self.label_int_mapping.keys():
                continue
            
            if dataset == 'TractoInferno':
                cls_path = os.path.join(data_path, subject, 'tractography', clss)
            elif dataset == '105HCP':
                cls_path = os.path.join(data_path, subject, 'tracts', clss)
            else:
                raise FileNotFoundError('No tractography or tracts folder found in subject directory')
            loaded_streamlines = load_stream(cls_path)

            if len(loaded_streamlines) > samples_per_class:
                loaded_streamlines = [loaded_streamlines[i] for i in np.random.choice(len(loaded_streamlines), samples_per_class, replace=False)]
            
            streamlines.extend(loaded_streamlines)
            streamline_to_class.update({self.hash_streamline(streamline): class_name for streamline in loaded_streamlines})

        if len(streamlines) > sample_per_subject:
            streamlines = [streamlines[i] for i in np.random.choice(len(streamlines), sample_per_subject, replace=False)]

        labels = [streamline_to_class[self.hash_streamline(streamline)] for streamline in streamlines]
        return subject_id, blist(streamlines), labels

    def load_streamlines_mp(self, data_paths):

        streamlines = []
        subject_ids = []
        labels = []
        path1 = data_paths[0]
        path2 = data_paths[1]

        subjects_path1 = [subject for subject in Path(path1).iterdir() if os.path.isdir(subject)]
        if self.subset=='test' and type(self.config.subject_ids) == list: # using only the given subjects
            subjects_path1 = [subject for subject in subjects_path1 if subject.name in self.config.subject_ids]
        subjects_path2 = [subject for subject in Path(path2).iterdir() if os.path.isdir(subject)]
        if self.subset=='test' and type(self.config.subject_ids) == list: # using only the given subjects
            subjects_path2 = [subject for subject in subjects_path2 if subject.name in self.config.subject_ids]

        subject_data_path1 = [(subject, path1, self.sample_per_subject, self.samples_per_class) for subject in subjects_path1]
        subject_data_path2 = [(subject, path2, self.sample_per_subject, self.samples_per_class) for subject in subjects_path2]
        subject_data = subject_data_path1 + subject_data_path2
        # Using multiprocessing to parallelize the loading
        with Pool(processes=os.cpu_count()) as pool:
            results = pool.map(self.load_subject, subject_data)

        for subject_id, subject_streamlines, subject_labels in results:
            streamlines.extend(subject_streamlines)
            subject_ids.extend([subject_id] * len(subject_streamlines))
            labels.extend(subject_labels)

        print(f'Total number of streamlines in {self.subset} set: {len(streamlines)}')
        self.streamlines = blist(streamlines)
        self.subject_ids = np.array(subject_ids)
        self.labels = np.array(labels)
        
        print(np.unique(subject_ids))
        if self.subset != 'test':
            os.makedirs(f'{self.save_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}', exist_ok=True)
            with open(f'{self.save_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/streamlines.pkl', 'wb') as f:
                pickle.dump(self.streamlines, f)
            with open(f'{self.save_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/labels.pkl', 'wb') as f:
                pickle.dump(self.labels,f)
            with open(f'{self.save_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/subjectIds.pkl', 'wb') as f:
                pickle.dump(self.subject_ids, f)
        
        del streamlines
        del subject_ids
        del labels
        gc.collect()

    def fix_pts_in_streamlines(self):
        # pts = self.points_per_streamline
        # streamlines, pts = args
        x = [(streamline, self.pts_per_sample) for streamline in self.streamlines]
        with Pool(processes=os.cpu_count()) as pool:
            resampled_streamlines = pool.map(choose_upORdownsample, x)
        resampled_streamlines = np.array(resampled_streamlines)

        if self.subset != 'test':
            with open(f'{self.save_path}/streamlineData_{self.subset}_{self.samples_per_class}_{self.sample_per_subject}/resampledStreamlines_{self.points_per_streamline}.pkl', 'wb') as f:
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
    
    def fix_streamline(self, args):
        streamline, pts, use_resampled = args
        streamline = np.array(streamline)
        pts_in_streamline = len(streamline)
        streamline_reps = np.tile(streamline, (pts//pts_in_streamline, 1))
        if pts%pts_in_streamline > 0:
            streamline_reps = np.concatenate((streamline_reps, streamline[np.random.choice(pts_in_streamline, pts%pts_in_streamline)]), axis=0)

        return self.pc_norm(streamline_reps)
    def __getitem__(self,index):
        if self.subset == 'test':
            return torch.from_numpy(np.array(self.label_int_mapping[self.labels[index]])), [self.subject_ids[index], str(self.streamlines[index])], torch.from_numpy(np.array(self.pc_norm(self.resampled_streamlines[index]), dtype=np.float32))
        elif not self.use_custom_batch:
            if self.config.use_resampled_streamlines:
                return torch.from_numpy(np.array(self.label_int_mapping[self.labels[index]])), [0], torch.from_numpy(self.pc_norm(self.resampled_streamlines[index]).astype(np.float32))
            else:
                return torch.from_numpy(np.array(self.label_int_mapping[self.labels[index]])), [0], torch.from_numpy(self.pc_norm(np.array(self.streamlines[index])))
        
        else:
            batch_size = self.custom_batches.shape[1]
            # streamlines = [self.streamlines[i] for i in self.custom_batches[index]]
            # streamlines = self.fix_pts_in_streamlines((streamlines, self.points_per_streamline))
            streamlines = self.resampled_streamlines[self.custom_batches[index]]
            # print(streamlines.shape)
            streamlines = np.array(list(map(self.pc_norm, streamlines)), dtype=np.float32)
            batch_labels = self.labels[self.custom_batches[index]]
            batch_labels = self.map_batch_labels(batch_labels)

            shuffle = np.random.permutation(batch_size)
            # print(streamlines.shape)
            # print(batch_labels.shape)
            # print(shuffle.shape)
            return torch.from_numpy(batch_labels[shuffle]), [0], torch.from_numpy(streamlines[shuffle])
    
    def __len__(self):
        if not self.use_custom_batch:
            return len(self.labels)
        else:
            return len(self.custom_batches)
    
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