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
sys.path.append('./TractoGPT/')
from utils.config import *
from utils.logger import *
from utils.qbx_clustering import create_qbx_cluster_from_scratch_n_save
import gc, pickle
import copy
from memory_profiler import profile
from .build import DATASETS

#%%
def create_clusters_subject(args):
    data, subject_id, min_threshold_per_cluster, test = args

    cluster_id_to_idx_mapping = {}
    cluster_id_to_idx_mapping['L1'] = {}
    for i in range(len(data['cluster_ids_l1'])):
        cluster_id_to_idx_mapping['L1'][data['cluster_ids_l1'][i]] = i
    cluster_id_to_idx_mapping['L2'] = {}
    for i in range(len(data['cluster_ids_l2'])):
        cluster_id_to_idx_mapping['L2'][data['cluster_ids_l2'][i]] = i
    cluster_id_to_idx_mapping['L3'] = {}
    for i in range(len(data['cluster_ids_l3'])):
        cluster_id_to_idx_mapping['L3'][data['cluster_ids_l3'][i]] = i
    
    clusters = []
    pc_cluster_labels = []
    included_streamlines = np.zeros(len(data['label'])) # none included here
    cluster_ids_l3, cluster_count_l3 = np.unique(data['cluster_ids_l3'], return_counts=True)

    cluster_ids_l2 = []
    for cluster_id in cluster_ids_l3[cluster_count_l3 <  min_threshold_per_cluster]:
        cluster_ids_l2.append(data['cluster_ids_l2'][cluster_id_to_idx_mapping['L3'][cluster_id]])
    cluster_ids_l2 = np.unique(cluster_ids_l2)
    cluster_count_l2 = np.zeros(len(cluster_ids_l2))
    for i,cluster_id in enumerate(cluster_ids_l2):
        cluster_count_l2[i] = len(np.where(data['cluster_ids_l2'] == cluster_id)[0])
    
    cluster_ids_l1 = []
    for cluster_id in cluster_ids_l2[cluster_count_l2 <  min_threshold_per_cluster]:
        cluster_ids_l1.append(data['cluster_ids_l1'][cluster_id_to_idx_mapping['L2'][cluster_id]])
    cluster_ids_l1 = np.unique(cluster_ids_l1)
    cluster_count_l1 = np.zeros(len(cluster_ids_l1))
    for i,cluster_id in enumerate(cluster_ids_l1):
        cluster_count_l1[i] = len(np.where(data['cluster_ids_l1'] == cluster_id)[0])
    
    for cluster_id in cluster_ids_l1:
        cluster_streamline_idx = np.where(data['cluster_ids_l1'] == cluster_id)[0]
        if included_streamlines[cluster_streamline_idx[0]] == 1:
            continue
        if test==False and len(cluster_streamline_idx)< min_threshold_per_cluster:
            continue
        included_streamlines[cluster_streamline_idx] = 1
        u_labels, u_counts = np.unique(data['label'][cluster_streamline_idx], return_counts=True)
        label = u_labels[np.argmax(u_counts)]
        cluster_streamline_idx = [f'{subject_id}_{i}' for i in cluster_streamline_idx]
        clusters.append(cluster_streamline_idx)
        pc_cluster_labels.append(label)

    for cluster_id in cluster_ids_l2:
        cluster_streamline_idx = np.where(data['cluster_ids_l2'] == cluster_id)[0]
        if included_streamlines[cluster_streamline_idx[0]] == 1:
            continue
        if test==False and len(cluster_streamline_idx)< min_threshold_per_cluster:
            continue
        included_streamlines[cluster_streamline_idx] = 1
        u_labels, u_counts = np.unique(data['label'][cluster_streamline_idx], return_counts=True)
        label = u_labels[np.argmax(u_counts)]
        cluster_streamline_idx = [f'{subject_id}_{i}' for i in cluster_streamline_idx]
        clusters.append(cluster_streamline_idx)
        pc_cluster_labels.append(label)

    for cluster_id in cluster_ids_l3:
        cluster_streamline_idx = np.where(data['cluster_ids_l3'] == cluster_id)[0]
        if included_streamlines[cluster_streamline_idx[0]] == 1:
            continue
        if test==False and len(cluster_streamline_idx)< min_threshold_per_cluster:
            continue
        included_streamlines[cluster_streamline_idx] = 1
        u_labels, u_counts = np.unique(data['label'][cluster_streamline_idx], return_counts=True)
        label = u_labels[np.argmax(u_counts)]
        cluster_streamline_idx = [f'{subject_id}_{i}' for i in cluster_streamline_idx]
        clusters.append(cluster_streamline_idx)
        pc_cluster_labels.append(label)

    return (clusters, pc_cluster_labels)
def load_data(args):
    pth, use_resampled_streamline, pts_per_sample, min_threshold_per_cluster, test = args
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
    del data_x
    del data

    cluster, pc_cluster_labels = create_clusters_subject((subject_data, subject_id, min_threshold_per_cluster, test))

    pc_cluster = np.zeros((len(cluster), pts_per_sample, 3))
    for i in range(len(cluster)):
        pc_cluster[i] = concat_streamlines((subject_data, cluster[i], use_resampled_streamline, pts_per_sample))
    
    cluster_labels = []
    for i in range(len(cluster)):
        streamline_idx = np.array([int(cluster[i][j].split('_')[1]) for j in range(len(cluster[i]))])
        cluster_labels.append(subject_data['label'][streamline_idx])
    if test:
        return (pc_cluster, pc_cluster_labels, cluster, cluster_labels, blist(subject_data['streamlines']), subject_data['label'])
    return (pc_cluster, pc_cluster_labels, cluster, cluster_labels)


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
    
    # club_streamlines = pc_norm(club_streamlines)
    original_club_streamlines = copy.deepcopy(club_streamlines)
    while len(club_streamlines)+len(original_club_streamlines) < pts_per_sample:
        club_streamlines = np.concatenate((club_streamlines,original_club_streamlines))

    if len(original_club_streamlines) > pts_per_sample:
        club_streamlines = original_club_streamlines[np.random.choice(len(original_club_streamlines), pts_per_sample, replace=False)]
    else:
        club_streamlines = np.concatenate((club_streamlines,original_club_streamlines[np.random.choice(len(original_club_streamlines), pts_per_sample-len(club_streamlines), replace=False)]))
    
    return np.array(club_streamlines)
@DATASETS.register_module()
class qbx_goback_3L(Dataset):

    def __init__(self,config):
        self.config = config
        self.raw_data_path = config.raw_data_path
        self.data_path = config.data_path
        self.subset = config.subset
        self.save_path = os.path.join(self.data_path, 'processed_data')
        self.data_path = os.path.join(self.data_path, self.subset)
        self.use_custom_batch = config.use_custom_batch
        self.pts_per_sample = config.pts_per_sample
        self.min_batches = config.minimum_batches
        self.samples_per_class_per_batch = config.samples_per_class_per_batch
        self.use_resampled_streamline = config.use_resampled_streamline
        self.min_threshold_per_cluster = config.min_threshold_per_cluster
        self.label_mapping_path = config.label_mapping_path
        if self.subset == 'validset' or self.subset == 'testset':
            self.use_custom_batch = False

        if self.use_resampled_streamline:
            raise NotImplementedError('Resampled Streamlines not implemented yet')
        self.label_int_mapping = self.load_label_mapping()
        if not os.path.exists(self.data_path):
            print_log(f'[DATASET] Data path {self.data_path} does not exist', logger='qbx_goback_3L')
            print_log(f'[DATASET] Creating QBX clusters', logger='qbx_goback_3L')
            create_qbx_cluster_from_scratch_n_save(
                dataset='TractoInferno',
                mapping=self.label_int_mapping,
                save_path='/'.join(self.data_path.split('/')[:-1]),
                raw_data_path=self.raw_data_path,
                subset=[self.subset]
            )

        print_log(f'[DATASET] Loading {self.subset} data', logger='qbx_goback_3L')
        if self.subset == 'testset' and os.path.exists(os.path.join(self.save_path, f'{self.subset}.pkl')):
            print_log(f'[DATASET] Loading Processed Data', logger='qbx_goback_3L')
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
            print_log(f'[DATASET] Loading Processed Data', logger='qbx_goback_3L')
            with open(os.path.join(self.save_path, f'{self.subset}.pkl'), 'rb') as f:
                self.pc_cluster, self.pc_cluster_labels, self.cluster, self.cluster_labels = pickle.load(f)
            self.cluster = blist(self.cluster)
            self.cluster_labels = blist(self.cluster_labels)
        else:
            print_log(f'[DATASET] Loading and processing data', logger='qbx_goback_3L')
            self.load_and_process_data()

        print_log(f'[DATASET] {len(self.pc_cluster)} samples loaded', logger='qbx_goback_3L')
        print_log(f'[DATASET] {len(np.unique(self.pc_cluster_labels, return_counts=True)[1])} unique classes', logger='qbx_goback_3L')
        if self.use_custom_batch:
            print_log(f'[DATASET] Building custom batches', logger='qbx_goback_3L')
            self.create_custom_batches()

    def load_label_mapping(self):
        with open(self.label_mapping_path, 'r') as f:
            label_mapping = json.load(f)
        
        return label_mapping

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

        # subjects = [subject for subject in Path(self.data_path).iterdir() if os.path.isfile(subject)]
        subjects = [subject for subject in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, subject))]
        print(subjects)
        print(self.config.subject_ids)
        if self.subset == 'testset' and type(self.config.subject_ids) == list: # using only the given subjects
            # subjects = [subject for subject in subjects if subject.name in self.config.subject_ids]
            subjects = [subject for subject in subjects if subject.split('.')[0] in self.config.subject_ids]
        print(subjects)
        print(subjects)
        print(subjects)
        print(subjects)
        print(subjects)
        inp = []
        for subject in subjects:
            inp.append((os.path.join(self.data_path,subject), self.use_resampled_streamline, self.pts_per_sample, self.min_threshold_per_cluster, self.subset == 'testset'))
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
            if self.subset == 'testset':
                self.streamlines[subjects[i].split('.')[0]] = data[i][4]
                self.streamline_labels[subjects[i].split('.')[0]] = data[i][5]
        
        self.pc_cluster = np.array(self.pc_cluster)
        self.pc_cluster_labels = np.array(self.pc_cluster_labels)
        
        os.makedirs(self.save_path, exist_ok=True)
        if self.subset != 'testset':
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
        if self.subset == 'testset':
            label = self.map_batch_labels([self.pc_cluster_labels[index]])
            pc = self.pc_norm(self.pc_cluster[index])
            subject = self.cluster[index][0].split('_')[0]
            streamline_idx = [int(i.split('_')[1]) for i in self.cluster[index]]
            streamlines = [list(self.streamlines[subject][i]) for i in streamline_idx]
            streamline_labels = self.streamline_labels[subject][streamline_idx]
            streamline_labels = list(self.map_batch_labels(streamline_labels))
            subject = [subject]*len(streamlines)
            return torch.tensor(label).long(), [str(subject), str(streamline_labels), str(streamlines)], torch.tensor(pc).float()
            # return torch.tensor(label).long(), [str(0), str(0), str(0)], torch.tensor(pc).float()


        if not self.use_custom_batch:
            label = self.map_batch_labels([self.pc_cluster_labels[index]])
            pc = self.pc_norm(self.pc_cluster[index])
            return torch.tensor(label).long(), [0], torch.tensor(pc).float()
        else:
            batch_label = self.map_batch_labels([self.pc_cluster_labels[i] for i in self.custom_batches[index]])
            batch_pc = np.array([self.pc_norm(self.pc_cluster[i]) for i in self.custom_batches[index]])

            shuffle_batch = np.random.choice(len(batch_label), len(batch_label), replace=False)
            batch_label = batch_label[shuffle_batch]
            batch_pc = batch_pc[shuffle_batch]

            return torch.from_numpy(batch_label.astype(np.float32)), [0], torch.from_numpy(batch_pc.astype(np.float32))
    def __len__(self):
        if not self.use_custom_batch:
            return len(self.pc_cluster)
        else:
            return len(self.custom_batches)
        
#%%
# config = cfg_from_yaml_file('./TractoGPT/cfgs/dataset_configs/qbx_goback_3L.yaml')
# dataset = qbx_goback_3L(config)
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
# # with open('./data/dataTrInF/qbx/validset/sub-1254.pkl', 'rb') as f:
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