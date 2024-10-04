#%%
from .build import DATASETS
import sys
# sys.path.append('../../')
# sys.path.append('../')
import datetime
# from utils import alpha
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset
# from knn_cuda import KNN
import torch
import os
import gc
import json
# from utils.funcs import cluster2tract_label,obtain_TractClusterMapping
from utils.config import *
from multiprocessing import Pool, cpu_count
# import faiss
# from memory_profiler import profile
#%%
@DATASETS.register_module()
class StreamlinePatchKnn(Dataset):

    # @profile
    def __init__(self, config):
        self.config = config
        self.path = config.path
        self.subset = config.subset

        if self.subset == 'extra_train': # handling extra_train subset in BERT pretraining
            file_name = 'val'
        else:
            file_name = self.subset

        self.data_path = os.path.join(self.path,file_name+".pickle")
        self.num_local_neighbours = config.n_local_neighbours
        self.points_per_streamline = int(config.points_per_streamline)

        self.data = self.get_streamlines() # reading the pickle file
        self.streamlines = np.array(self.data['feat'])
        # self.data['subject_id'] = self.data['subject_id'][:10000]
        self.labels = np.array(self.data['label'])
        # self.save_class_freq()

        self.resampled_streamlines = np.array(self.interpolate_streamline()) # increased points in streamlines
        # self.resampled_streamlines = self.streamlines

        print('interpolation completed')
        self.neighbour_indices = np.array(self.get_local_streamlines_subjectwise_CD())
        print('neighbours completed')
        self.itr_cnt = 0

        del self.streamlines
        del self.data
        gc.collect()

    # def save_class_freq(self):
    #     clusters = np.arange(1600)
    #     mappings = obtain_TractClusterMapping('/scratch/jankita.scee.iitmandi/TractoBERT/FiberClusterAnnotation_Updated20230110.xlsx')

    #     tracts = cluster2tract_label(clusters,mappings, False)
    #     tracts = np.array(tracts)

    #     class_freq = dict()
    #     for label in self.labels:
    #         if tracts[label] not in class_freq.keys():
    #             class_freq[tracts[label]] = 0
    #         class_freq[tracts[label]]+=1
    #     class_freq = {int(k): int(v) for k, v in class_freq.items()}

    #     with open(f'experiments/class_frequency_{self.subset}.json', "w") as f:
    #         json.dump(class_freq, f)

    def interp_streamline(self,streamline):
        old_points = np.linspace(0, 1, streamline.shape[0])
        new_points = np.linspace(0, 1, self.points_per_streamline)
        
        # Interpolation functions for x, y, z
        x_interp = CubicSpline(old_points, streamline[:, 0], bc_type='natural')
        y_interp = CubicSpline(old_points, streamline[:, 1], bc_type='natural')
        z_interp = CubicSpline(old_points, streamline[:, 2], bc_type='natural')
        
        # Resample streamline
        resampled_streamline = np.vstack([
            x_interp(new_points),
            y_interp(new_points),
            z_interp(new_points)
        ]).T
        return resampled_streamline
    def interpolate_streamline(self):
        a,b,c = np.shape(self.streamlines)
        # xyz = [(j,i) for i,j in enumerate(self.streamlines)]
        # print('start')
        print(datetime.datetime.now())
        with Pool(processes=cpu_count()) as pool:
            resampled_streamlines = pool.map(self.interp_streamline, self.streamlines)

        resampled_streamlines = np.array(resampled_streamlines)
        print(datetime.datetime.now())
        # print(resampled_streamlines)
        return resampled_streamlines

    def get_local_streamlines(self, data):
        """finds "n" local streamlines for every streamline"""
        data_reshaped = data.reshape(data.shape[0], -1)

        # Number of neighbors to find for each streamline
        n_neighbors = self.num_local_neighbours

        # Create a NearestNeighbors instance
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(data_reshaped)

        # Find the nearest neighbors for each streamline
        distances, indices = nbrs.kneighbors(data_reshaped)

        return indices
    
    def get_local_streamlines_subjectwise(self):
        # Reshape the data to (N, 45)
        # data_reshaped = data.reshape(data.shape[0], -1)

        num_streamlines = self.streamlines.shape[0]
        all_indices = []

        i=0
        # Process data in batches
        for subject_id in np.unique(self.data['subject_id']):

            batch = self.streamlines[self.data['subject_id'] == subject_id]
            batch = batch.reshape(len(batch),-1)
            # Create and fit a NearestNeighbors instance on the current batch
            nbrs = NearestNeighbors(n_neighbors=self.num_local_neighbours, algorithm='auto')
            nbrs.fit(batch)

            # Find the nearest neighbors for each streamline in the batch
            _, batch_indices = nbrs.kneighbors(batch)

            # Adjust indices to be relative to the batch
            batch_indices += i  # Offset indices to reflect the overall dataset
            i+=len(batch)

            all_indices.append(batch_indices)

        # Concatenate all the indices from the batches
        indices = np.vstack(all_indices)

        return indices

    def get_local_streamlines_cuda(self, data):
        # Convert NumPy array to PyTorch tensor
        # data_tensor = torch.from_numpy(data).float().to('cuda')

        # Reshape the data to (N, 45)
        data_reshaped = data.reshape(data.shape[0], -1)  # Shape: (N, 45)

        num_streamlines = data_reshaped.shape[0]
        indices_list = []

        for i in range(0, num_streamlines, self.batch_size):
            end_i = min(i + self.batch_size, num_streamlines)
            batch = torch.from_numpy(data_reshaped[i:end_i]).float().to('cuda')

            # Compute pairwise squared distances for the batch against the whole dataset
            distances = torch.cdist(batch, batch, p=2)

            # Get the indices of the nearest neighbors
            _, batch_indices = torch.topk(distances, self.num_local_neighbours, largest=False)

            indices_list.append(batch_indices)

        # Concatenate all the indices from the batches
        indices = torch.cat(indices_list, dim=0)

        return indices.cpu()

    def chamfer_distance(self,x, y):
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
        
        return chamfer_dist

    def get_local_streamlines_subjectwise_CD(self):
        """  finding local streamlines using chamfer distance (cuda implimentation)"""
        all_indices = []

        i = 0
        for subject_id in np.unique(self.data['subject_id']):
            """ creating a batch for each subject individually """
            batch = self.streamlines[self.data['subject_id'] == subject_id]
            batch = torch.tensor(batch, device='cuda')

            # Compute Chamfer distance matrix using GPU
            num_streamlines = batch.size(0)
            chamfer_matrix = torch.zeros((num_streamlines, num_streamlines), device='cuda')

            for j in range(num_streamlines):
                # if(i%10000 == 0): print(i)
                chamfer_matrix[j, :] = self.chamfer_distance(batch[j].unsqueeze(0), batch)

            # Use the Chamfer distance matrix in NearestNeighbors with precomputed metric
            chamfer_matrix = chamfer_matrix.cpu().numpy()  # Move matrix back to CPU for sklearn
            nbrs = NearestNeighbors(n_neighbors=self.num_local_neighbours, metric='precomputed')
            nbrs.fit(chamfer_matrix)

            # Find the nearest neighbors for each streamline in the batch
            _, batch_indices = nbrs.kneighbors(chamfer_matrix)

            # Adjust indices to be relative to the batch
            batch_indices += i  # Offset indices to reflect the overall dataset
            i += len(batch)

            all_indices.append(batch_indices)
            del batch
            gc.collect()

        # Concatenate all the indices from the batches
        indices = np.vstack(all_indices)

        return indices

    def get_streamlines(self):
        """reads data from the given pickle file"""
        with open(self.data_path, 'rb') as file:
            data = pickle.load(file)
        return data
    
    def pc_norm(self, pc):
        """normalising the point cloud"""
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    
    # def get_tract_label(self, cluster_label):

    #     ordered_tract_cluster_mapping_dict = obtain_TractClusterMapping('FiberClusterAnnotation_Updated20230110.xlsx')  # {'tract name': ['cluster_xxx','cluster_xxx', ... 'cluster_xxx']} 
    #     cur_label = cluster2tract_label(cluster_label, ordered_tract_cluster_mapping_dict, output_lst=False)
    #     # print(cur_label)
    #     return cur_label

    def __getitem__(self,index):

        if self.subset == 'extra_train':
            index = len(self.neighbour_indices) - index - 1
        
        streamlines = self.resampled_streamlines[self.neighbour_indices[index]]
        main_streamline_index = self.neighbour_indices[index][0]
        label = self.labels[main_streamline_index]

        streamlines = streamlines.reshape(self.num_local_neighbours*self.points_per_streamline, 3)
        streamlines = self.pc_norm(streamlines)

        # tract_label = self.get_tract_label(label)
        # print(tract_label)
        return label , 0 , torch.from_numpy(streamlines.astype(np.float32))

    def  __len__(self):
        if self.subset == 'val' or self.subset == 'extra_train':
            """first half samples for validation and second half samples for extra_train required by BERT"""
            return int(len(self.neighbour_indices)//2)
        return len(self.neighbour_indices)
        
#%%
# if __name__ == "__main__":
#     import sys 
#     sys.path.append("../../")
#     config_path = "../cfgs/dataset_configs/streamline_patch.yaml"
#     config = cfg_from_yaml_file(config_path)
#     dataset = StreamlinePatchOld(config)

# # %%
# pcd = dataset[1000][2]
# import plotly.graph_objects as go

# fig = go.Figure()

# x_values = pcd[:config.points_per_streamline, 0]  # X-axis values
# y_values = pcd[:config.points_per_streamline, 1]  # Y-axis values
# z_values = pcd[:config.points_per_streamline, 2]  # Z-axis values

# # Add the line to the plot
# fig.add_trace(go.Scatter3d(x=x_values, y=y_values, z=z_values, 
#                             mode='markers', marker=dict(size=2,color = 'red')))

# x_values = pcd[config.points_per_streamline:, 0]  # X-axis values
# y_values = pcd[config.points_per_streamline:, 1]  # Y-axis values
# z_values = pcd[config.points_per_streamline:, 2]  # Z-axis values

# # Add the line to the plot
# fig.add_trace(go.Scatter3d(x=x_values, y=y_values, z=z_values, 
#                             mode='markers', marker=dict(size=2,color = 'blue')))

# # Update the layout
# fig.update_layout(
#     title='3D Plot of Multiple Lines',
#     scene=dict(
#         xaxis_title='X-axis',
#         yaxis_title='Y-axis',
#         zaxis_title='Z-axis'
#     ),
#     legend_title='Lines'
# )

# # Show the plot
# fig.show()
# # %%
