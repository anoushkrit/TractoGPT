# %%
import sys
import os,json
import datetime, time
import numpy as np
from pathlib import Path
import torch
from blist import blist
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.interpolate import CubicSpline
from multiprocessing import Pool, cpu_count, set_start_method
from dipy.tracking.distances import approx_polygon_track as apt
sys.path.append('./TractoGPT/')
from utils.config import *
from utils.logger import *
from utils.qbx_clustering import create_qbx_cluster_from_scratch_n_save
import gc, pickle
import copy
from memory_profiler import profile
from collections import defaultdict
import tracemalloc
from .build import DATASETS
# tracemalloc.start()
#%%
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
    # current, peak = tracemalloc.get_traced_memory()
    # print(f'current memory usage at start of load_data is {current / 10**6}MB; Peak was {peak / 10**6}MB')
    pth, use_resampled_streamline, pts_per_sample, streamline_threshold_per_cluster, test, subject, pts_per_streamline, save_path, subset = args
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
    # print(2)

    cluster, pc_cluster_labels = create_clusters_subject((subject_data, subject_id, streamline_threshold_per_cluster, test))
    subject_data['cluster_ids_l1'] = None
    subject_data['cluster_ids_l2'] = None
    subject_data['cluster_ids_l3'] = None
    # print(3)

    pc_cluster = np.zeros((len(cluster), pts_per_sample, 3))
    for i in range(len(cluster)):
        pc_cluster[i] = concat_streamlines((subject_data, cluster[i], use_resampled_streamline, pts_per_sample))
    
    cluster_labels = []
    for i in range(len(cluster)):
        streamline_idx = np.array([int(cluster[i][j].split('_')[1]) for j in range(len(cluster[i]))])
        cluster_labels.append(subject_data['label'][streamline_idx])

    # resampling streamline
    output_dir = os.path.join(save_path, f'resampled_streamlines_{pts_per_streamline}', subset)
    os.makedirs(output_dir, exist_ok=True)
    inp = [(subject_data['streamlines'][i], pts_per_streamline) for j in range(len(subject_data['streamlines']))]
    with Pool(20) as p:
        resampled_streamlines = p.map(choose_upORdownsample, inp)
    
    resampled_streamlines = np.array(resampled_streamlines)
    # Directly write to memmap
    filename = os.path.join(output_dir, f'{subject.split(".")[0]}__{len(resampled_streamlines)}.dat')
    memmap = np.memmap(filename, dtype='float32', mode='w+', shape=(resampled_streamlines.shape))
    memmap[:] = np.array(resampled_streamlines)  # Reshape if necessary
    memmap.flush()
    del resampled_streamlines   

    subject_labels = subject_data['label']
    streamlines = subject_data['streamlines']
    del subject_data
    gc.collect()
    # current, peak = tracemalloc.get_traced_memory()
    # print(f'current memory usage at end of load_data is {current / 10**6}MB; Peak was {peak / 10**6}MB')
    # return (pc_cluster, pc_cluster_labels, cluster, cluster_labels, None, blist(subject_data['label']))
    if test:
        return (pc_cluster, pc_cluster_labels, cluster, None, streamlines, subject_labels)
    return (pc_cluster, pc_cluster_labels, cluster, None, None, subject_labels)
    # return (pc_cluster, pc_cluster_labels, cluster, cluster_labels)

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
class goback_3L_streamline_TrInF(Dataset):

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
        self.pc_samples_per_class_per_batch = config.pc_samples_per_class_per_batch
        self.fusion_streamline_per_cluster = config.fusion_streamline_per_cluster
        self.use_resampled_streamline = config.use_resampled_streamline
        self.streamline_threshold_per_cluster = config.streamline_threshold_per_cluster
        self.label_mapping_path = config.label_mapping_path

        if self.streamline_threshold_per_cluster < self.fusion_streamline_per_cluster:
            raise ValueError('streamline_threshold_per_cluster should be greater than or equal to fusion_streamline_per_cluster')
        
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

        os.makedirs(self.save_path, exist_ok=True)

        if self.subset == 'testset' and os.path.exists(os.path.join(self.save_path, f'{self.subset}.pkl')):
            print_log(f'Loading {self.subset} from {self.save_path}', logger='goback_3L_streamline_TrInF')
            with open(os.path.join(self.save_path, f'{self.subset}.pkl'), 'rb') as f:
                self.pc_cluster, self.pc_cluster_labels, self.cluster, self.cluster_labels, self.streamlines, self.streamline_labels= pickle.load(f)
                for i in self.streamlines.keys():
                    self.streamlines[i] = blist(self.streamlines[i])
            
            if type(self.config.subject_ids) == list:
                print_log(f'Using only the given subjects', logger='goback_3L_streamline_TrInF')
                self.pc_cluster = [self.pc_cluster[i] for i in range(len(self.pc_cluster)) if self.cluster[i][0].split('_')[0] in self.config.subject_ids]
                self.pc_cluster_labels = [self.pc_cluster_labels[i] for i in range(len(self.pc_cluster_labels)) if self.cluster[i][0].split('_')[0] in self.config.subject_ids]
                self.cluster_labels = [self.cluster_labels[i] for i in range(len(self.cluster_labels)) if self.cluster[i][0].split('_')[0] in self.config.subject_ids]
                self.streamlines = {i: self.streamlines[i] for i in self.streamlines.keys() if i in self.config.subject_ids}
                self.streamline_labels = {i: self.streamline_labels[i] for i in self.streamline_labels.keys() if i in self.config.subject_ids}
                self.cluster = [self.cluster[i] for i in range(len(self.cluster)) if self.cluster[i][0].split('_')[0] in self.config.subject_ids]
                self.pc_cluster = np.array(self.pc_cluster)
                self.pc_cluster_labels = np.array(self.pc_cluster_labels)
            
            self.load_resampled_memmaps()
            streamline_count = 0
            for i in self.streamlines.keys():
                streamline_count += len(self.streamlines[i])

            self.streamline_to_cluster = np.zeros((streamline_count,2), dtype=np.int64)
            start=0
            for i, cluster in enumerate(self.cluster):
                for j in range(len(cluster)):
                    self.streamline_to_cluster[start+j] = [i,j]
                start += len(cluster)
            
            if(start!=streamline_count):
                print(start, streamline_count)


        elif os.path.exists(os.path.join(self.save_path, f'{self.subset}.pkl')):
            print_log(f'Loading {self.subset} from {self.save_path}', logger='goback_3L_streamline_TrInF')
            with open(os.path.join(self.save_path, f'{self.subset}.pkl'), 'rb') as f:
                data = pickle.load(f)
                if len(data) != 5:
                    del data
                    print_log(f'Loaded Data is not correct. Processing again', logger='goback_3L_streamline_TrInF')
                    self.load_and_process_data()
                else:
                    self.pc_cluster, self.pc_cluster_labels, self.cluster, self.cluster_labels, self.streamline_labels = data
            
            self.load_resampled_memmaps()
        else:
            self.load_and_process_data()

        print_log(f'Loaded {len(self.pc_cluster)} samples', logger='goback_3L_streamline_TrInF')
        if self.use_custom_batch:
            print_log(f'[DATASET] Building custom batches', logger='goback_3L_streamline_TrInF')
            self.create_custom_batches()
    
    def load_resampled_memmaps(self):
        base_path = os.path.join(self.save_path, f'resampled_streamlines_{self.config.pts_per_streamline}', self.subset)
        if not os.path.exists(base_path):
            self.make_save_resampled_streamlines()
        print_log(f'Loading resampled streamlines from {base_path}', logger='goback_3L_streamline_TrInF')
        subjects = os.listdir(base_path)
        
        self.resampled_streamlines = {}
        pts_per_streamline = self.config.pts_per_streamline

        for subject in subjects:
            filename = os.path.join(base_path, subject)
            pts = int(subject.split("__")[1].split(".")[0])
            self.resampled_streamlines[subject.split("__")[0]] = np.memmap(filename, dtype='float32', mode='r', shape=(pts, pts_per_streamline, 3))

    def load_label_mapping(self):
        with open(self.label_mapping_path, 'r') as f:
            label_mapping = json.load(f)
        
        return label_mapping

    def create_custom_batches(self):
        unique_labels, label_counts = np.unique(self.pc_cluster_labels, return_counts=True)
        batch_size = self.pc_samples_per_class_per_batch*len(unique_labels)*self.fusion_streamline_per_cluster
        num_batches = max(int(min(label_counts)//self.pc_samples_per_class_per_batch), self.min_batches)
        self.custom_batches = np.zeros((num_batches, batch_size, 2), dtype=int)
        samples_per_class_per_batch = self.pc_samples_per_class_per_batch*self.fusion_streamline_per_cluster

        for i, label in enumerate(unique_labels):
            label_indices = np.where(self.pc_cluster_labels == label)[0]
            num_samples = self.pc_samples_per_class_per_batch*num_batches
            full_rounds = num_samples//len(label_indices)
            remainder = num_samples%len(label_indices)
            new_label_indices = np.tile(label_indices, full_rounds)
            if remainder > 0:
                new_label_indices = np.concatenate((new_label_indices, np.random.choice(label_indices, remainder, replace=False)))
            np.random.shuffle(label_indices)

            fusion_streamline_indices = np.array([
                np.random.choice(self.cluster[i], self.fusion_streamline_per_cluster, replace=False) 
                for i in new_label_indices
            ])
            fusion_streamline_indices = np.array(
                [[int(i.split('_')[1]) for i in cluster] for cluster in fusion_streamline_indices]
            )

            new_label_indices = np.tile(new_label_indices, (self.fusion_streamline_per_cluster,1)).T
            new_label_indices = np.stack((new_label_indices.flatten(), fusion_streamline_indices.flatten()), axis=1)

            label_indices = new_label_indices.reshape((num_batches, samples_per_class_per_batch, 2))
            self.custom_batches[:, i*samples_per_class_per_batch : (i+1)*samples_per_class_per_batch] = label_indices

    def make_save_resampled_streamlines(self, args):
        data, subjects = args

        self.resampled_streamlines = {}
        output_dir = os.path.join(self.save_path, f'resampled_streamlines_{self.config.pts_per_streamline}', self.subset)
        os.makedirs(output_dir, exist_ok=True)

        # Create the pool outside of the loop
        print_log(f'[DATASET] Resampling streamlines', logger='goback_3L_streamline_TrInF')
        with Pool(5) as p:
            for i, subject in enumerate(subjects):
                inp = [(data[i][4][j], self.config.pts_per_streamline) for j in range(len(data[i][4]))]
                resampled_streamlines = p.map(choose_upORdownsample, inp)
                
                resampled_streamlines = np.array(resampled_streamlines)
                # Directly write to memmap
                filename = os.path.join(output_dir, f'{subject.split(".")[0]}__{len(resampled_streamlines)}.dat')
                memmap = np.memmap(filename, dtype='float32', mode='w+', shape=(resampled_streamlines.shape))
                memmap[:] = np.array(resampled_streamlines)  # Reshape if necessary
                memmap.flush()

                self.resampled_streamlines[subject.split('.')[0]] = memmap
        print_log(f'[DATASET] Resampled streamlines saved to {output_dir}', logger='goback_3L_streamline_TrInF')
        gc.collect()

    def load_and_process_data(self):

        subjects = [subject for subject in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, subject))]
        if self.subset == 'testset' and type(self.config.subject_ids) == list: # using only the given subjects
            subjects = [subject for subject in subjects if subject.split('.')[0] in self.config.subject_ids]
            
        inp = []
        for subject in subjects:
            inp.append((os.path.join(self.data_path,subject), 
                        self.use_resampled_streamline,
                        self.pts_per_sample,
                        self.streamline_threshold_per_cluster,
                        self.subset == 'testset',
                        subject,
                        self.config.pts_per_streamline,
                        self.save_path,
                        self.subset
                        ))
        # with Pool(5) as p:
        #     # data = p.map(load_data, inp)
        #     data = list(tqdm(p.imap(load_data, inp), total=len(inp)))
        
        self.pc_cluster = []
        self.pc_cluster_labels = []
        self.cluster = []
        self.cluster_labels = []
        self.streamlines = {}
        self.streamline_labels = {}
        for i in tqdm(range(len(inp))):
            data = load_data(inp[i])
            self.pc_cluster.extend(data[0])
            self.pc_cluster_labels.extend(data[1])
            self.cluster.extend(data[2])
            if data[3] is not None:
                self.cluster_labels.extend(data[3])
            self.streamline_labels[subjects[i].split('.')[0]] = data[5]
            if self.subset == 'testset':
                self.streamlines[subjects[i].split('.')[0]] = data[4]
            # current, peak = tracemalloc.get_traced_memory()
            # print(f'Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB')

        # for i in range(len(data)):
        #     self.pc_cluster.extend(data[i][0])
        #     self.pc_cluster_labels.extend(data[i][1])
        #     self.cluster.extend(data[i][2])
        #     self.cluster_labels.extend(data[i][3])
        #     self.streamline_labels[subjects[i].split('.')[0]] = data[i][5]
        #     if self.subset == 'testset':
        #         self.streamlines[subjects[i].split('.')[0]] = data[i][4]
        

        self.pc_cluster = np.array(self.pc_cluster)
        self.pc_cluster_labels = np.array(self.pc_cluster_labels)
                
        self.load_resampled_memmaps()
        if self.subset != 'testset':
            with open(os.path.join(self.save_path, f'{self.subset}.pkl'), 'wb') as f:
                pickle.dump((self.pc_cluster, self.pc_cluster_labels, self.cluster, self.cluster_labels, self.streamline_labels), f)
        else:
            streamline_count = 0
            for i in self.streamlines.keys():
                streamline_count += len(self.streamlines[i])

            self.streamline_to_cluster = np.zeros((streamline_count,2), dtype=np.int64)
            start=0
            for i, cluster in enumerate(self.cluster):
                for j in range(len(cluster)):
                    self.streamline_to_cluster[start+j] = [i,j]
                start += len(cluster)
            
            if(start!=streamline_count):
                print(start, streamline_count)
        del data
        gc.collect()


    def map_batch_labels(self, labels):
        int_labels = [int(self.label_int_mapping[class_label]) for class_label in labels]
        return np.array(int_labels)
    
    def create_fusion_batch(self, args):
        batch_pc, batch_streamlines = args
        pc_pts = batch_pc.shape[1]-batch_streamlines.shape[1]

        random_pts = np.random.choice(batch_pc.shape[1], pc_pts, replace=False)
        batch_pc = batch_pc[:, random_pts] 
        
        batch_fusion = np.concatenate((batch_pc, batch_streamlines), axis=1)
        return batch_fusion
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    def __getitem__(self, index):
        if self.subset == 'testset':
            cluster_idx, cluster_streamline_idx = self.streamline_to_cluster[index]


            pc = self.pc_cluster[cluster_idx]
            subject = self.cluster[cluster_idx][0].split('_')[0]
            streamline_idx = int(self.cluster[cluster_idx][cluster_streamline_idx].split("_")[1])
            resampled_streamline = self.resampled_streamlines[subject][streamline_idx]
            streamline = self.streamlines[subject][streamline_idx]
            label = self.label_int_mapping[self.streamline_labels[subject][streamline_idx]]

            fusion = self.create_fusion_batch((np.expand_dims(pc, 0), np.expand_dims(resampled_streamline, 0)))[0]
            fusion = self.pc_norm(fusion)

            return torch.tensor(label).long(), [str(subject), str(streamline)], torch.tensor(fusion).float()
            # return torch.tensor(label).long(), [str(0), str(0), str(0)], torch.tensor(pc).float()


        elif not self.use_custom_batch:
            index = index//self.fusion_streamline_per_cluster
            pc = self.pc_cluster[index]
            subject = self.cluster[index][0].split('_')[0]
            streamline_idx = np.random.choice(self.cluster[index], 1)[0].split("_")[1]
            streamline = self.resampled_streamlines[subject][int(streamline_idx)]
            label = self.streamline_labels[subject][int(streamline_idx)]
            label = self.map_batch_labels([label])

            fusion = self.create_fusion_batch((np.expand_dims(pc, 0), np.expand_dims(streamline, 0)))[0]

            fusion = self.pc_norm(fusion)

            return torch.tensor(label).long(), [0], torch.tensor(fusion).float()

        else:
            batch_pc = []
            batch_streamlines = []
            batch_labels = []
            for i, cluster_idx in enumerate(self.custom_batches[index][:,0]):
                batch_pc.append(self.pc_cluster[cluster_idx])
                subject = self.cluster[cluster_idx][0].split('_')[0]
                # streamline_idx = int(np.random.choice(self.cluster[cluster_idx], 1)[0].split("_")[1])
                streamline_idx = self.custom_batches[index][i][1]
                batch_streamlines.append(self.resampled_streamlines[subject][streamline_idx])
                batch_labels.append(self.streamline_labels[subject][streamline_idx])

            batch_pc = np.array(batch_pc)
            batch_streamlines = np.array(batch_streamlines)
            batch_labels  = np.array(batch_labels)
            batch_labels = self.map_batch_labels(batch_labels)
            batch_fusion = self.create_fusion_batch((batch_pc, batch_streamlines))
            batch_fusion = np.array(list(map(self.pc_norm, batch_fusion)))

            shuffle_batch = np.random.choice(len(batch_labels), len(batch_labels), replace=False)

            batch_labels = batch_labels[shuffle_batch]
            batch_fusion = batch_fusion[shuffle_batch]

            return torch.from_numpy(batch_labels), [0], torch.from_numpy(batch_fusion.astype(np.float32))
    def __len__(self):
        if self.subset == 'testset':
            streamline_count = 0
            for i in self.streamlines.keys():
                streamline_count += len(self.streamlines[i])
            return streamline_count
        elif not self.use_custom_batch:
            return len(self.pc_cluster)*self.fusion_streamline_per_cluster
        else:
            return len(self.custom_batches)
        
#%%
# config = cfg_from_yaml_file('./TractoGPT/cfgs/dataset_configs/goback_3L_streamline_TrInF_23.yaml')
# dataset = goback_3L_streamline_TrInF(config)
# import time
# print('sleeping')
# time.sleep(10000000)
#%%
# label, streamlines, pc = dataset[1]
# #%%
# print(label.shape, pc.shape)
# import plotly.graph_objects as go
# import numpy as np

# def plot_point_cloud(points, title="3D Point Cloud"):
#     """
#     Plots a 3D point cloud using Plotly.

#     Args:
#         points (numpy.ndarray): A NumPy array of shape (N, 3) representing N points in 3D space.
#         title (str): Title of the plot.

#     Returns:
#         None. Displays the 3D plot.
#     """
#     # Ensure points are 3D
#     assert points.shape[1] == 3, "Points array must have shape (N, 3)"

#     # Create a scatter plot
#     scatter = go.Scatter3d(
#         x=points[:, 0],
#         y=points[:, 1],
#         z=points[:, 2],
#         mode='markers',
#         marker=dict(
#             size=1,
#             color='blue',  # You can choose a different color
#             opacity=0.8
#         )
#     )

#     # Set up the layout
#     layout = go.Layout(
#         title=title,
#         scene=dict(
#             xaxis_title='X',
#             yaxis_title='Y',
#             zaxis_title='Z'
#         ),
#         margin=dict(l=0, r=0, b=0, t=30)
#     )

#     # Create the figure and display it
#     fig = go.Figure(data=[scatter], layout=layout)
#     i = np.random.choice(20, 1)[0]
#     fig.write_html(f"fusion_plots_TrInF/{title}__{i}.html")

# #%%
# plot_point_cloud(pc[8], title = int(label[11]))
# #%%
# import json 
# with open('./TractoGPT/label_int_mappings/label_int_mapping_TrInF.json', 'r') as f:
#     label_mapping = json.load(f)
# int_label_mapping = {v: k for k, v in label_mapping.items()}
# print(int_label_mapping)
# #%%
# os.makedirs('fusion_plots_TrInF', exist_ok=True)
# for i in range(26):
#     label, _, pc = dataset[i]
#     for j in range(len(pc)):
#         plot_point_cloud(pc[j], title = int_label_mapping[int(label[j])])