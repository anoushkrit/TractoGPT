import torch
import torch.nn as nn
import torch.nn.functional as F
import timm, os
from torch.autograd import Variable
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import plotly.graph_objects as go
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from models.GPT import GPT_extractor, GPT_generator
import math
from models.z_order import *
import similaritymeasures


# class FrechetLoss(torch.nn.Module):
#     def __init__(self):
#         super(FrechetLoss, self).__init__()

#     def forward(self, generated, ground_truth):

        
#         # Compute mean and covariance for generated point clouds
#         mu_g = torch.mean(generated, dim=1, keepdim=True)  # Shape: (batch_size, 1, 3)
#         centered_g = generated - mu_g  # Centered points
#         cov_g = torch.bmm(centered_g.transpose(1, 2), centered_g) / (generated.size(1) - 1)  # Shape: (batch_size, 3, 3)

#         # Compute mean and covariance for ground truth point clouds
#         mu_gt = torch.mean(ground_truth, dim=1, keepdim=True)  # Shape: (batch_size, 1, 3)
#         centered_gt = ground_truth - mu_gt  # Centered points
#         cov_gt = torch.bmm(centered_gt.transpose(1, 2), centered_gt) / (ground_truth.size(1) - 1)  # Shape: (batch_size, 3, 3)

#         mu_diff = mu_g - mu_gt  # Shape: (batch_size, 1, 3)
#         mu_diff_sq = torch.norm(mu_diff, dim=2).squeeze() ** 2  # Shape: (batch_size,)

#         # Covariance distance
#         cov_g_inv = torch.linalg.inv(cov_g)  # Shape: (batch_size, 3, 3)
#         cov_product = torch.bmm(cov_g_inv, cov_gt)  # Shape: (batch_size, 3, 3)
        
#         cov_product_sqrt = torch.linalg.cholesky(cov_product + 1e-3 * torch.eye(3, device=cov_product.device).unsqueeze(0))  # Shape: (batch_size, 3, 3)

#         trace_term = torch.trace(cov_g + cov_gt - 2 * cov_product_sqrt)  # Shape: (batch_size,)

#         frechet_distance = mu_diff_sq + trace_term
#         return torch.mean(frechet_distance)

class FrechetLoss(torch.nn.Module):
    def __init__(self):
        super(FrechetLoss, self).__init__()
    
    def minkowski_distance(self, exp_data, num_data, p):
        # exp_data: (m, d)
        # num_data: (n, d)

        # Compute pairwise Minkowski distance
        # exp_data[:, None, :] creates (m, 1, d), num_data[None, :, :] creates (1, n, d)
        diff = torch.abs(exp_data[:, None, :] - num_data[None, :, :])  # Shape: (m, n, d)
        dist = torch.sum(diff**p, dim=2) ** (1/p)  # Minkowski distance, Shape: (m, n)

        return dist
    def forward(self, generated, ground_truth):
        """
        code copied from similaritymeasures library
        frechet_dist

        n = len(exp_data)
        m = len(num_data)
        c = distance.cdist(exp_data, num_data, metric='minkowski', p=p)
        ca = np.ones((n, m))
        ca = np.multiply(ca, -1)
        ca[0, 0] = c[0, 0]
        for i in range(1, n):
            ca[i, 0] = max(ca[i-1, 0], c[i, 0])
        for j in range(1, m):
            ca[0, j] = max(ca[0, j-1], c[0, j])
        for i in range(1, n):
            for j in range(1, m):
                ca[i, j] = max(min(ca[i-1, j], ca[i, j-1], ca[i-1, j-1]),
                            c[i, j])
        return ca[n-1, m-1]
        """
        B = generated.size(0)
        frechet_distance = torch.zeros(B, device=generated.device)
        for b in range(B):

            exp_data = generated[b]
            num_data = ground_truth[b]
            print(exp_data.shape, num_data.shape)
            n = len(exp_data)
            m = len(num_data)
            c = self.minkowski_distance(exp_data, num_data, 2)
            ca = torch.ones((n, m), device=generated.device)
            ca = torch.mul(ca, -1)
            ca[0, 0] = c[0, 0]
            for i in range(1, n):
                ca[i, 0] = max(ca[i-1, 0], c[i, 0])
            for j in range(1, m):
                ca[0, j] = max(ca[0, j-1], c[0, j])
            for i in range(1, n):
                for j in range(1, m):
                    ca[i, j] = max(min(ca[i-1, j], ca[i, j-1], ca[i-1, j-1]),
                                c[i, j])
            frechet_distance[b] = ca[n-1, m-1]
        
        return torch.mean(frechet_distance)

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class PointNetEncoder(nn.Module):
    def __init__(self,encoder_channel ,global_feat=True):
        super(PointNetEncoder, self).__init__()
        self.encoder_channel = encoder_channel
        self.stn = STN3d(3)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.encoder_channel, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.encoder_channel)
        self.global_feat = global_feat


    def forward(self, x):
        """
            points = B N 3
            output = B enc_dim    
        """
        x = x.transpose(2, 1)

        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)

        x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.encoder_channel)
       
        return x

class Encoder_large(nn.Module):  # Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(2048, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class Encoder_small(nn.Module):  # Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)
        self.knn_2 = KNN(k=1, transpose_mode=True)

    def simplied_morton_sorting(self, xyz, center):
        '''
        Simplifying the Morton code sorting to iterate and set the nearest patch to the last patch as the next patch, we found this to be more efficient.
        '''
        batch_size, num_points, _ = xyz.shape
        distances_batch = torch.cdist(center, center)
        distances_batch[:, torch.eye(self.num_group).bool()] = float("inf")
        idx_base = torch.arange(
            0, batch_size, device=xyz.device) * self.num_group
        sorted_indices_list = []
        sorted_indices_list.append(idx_base)
        distances_batch = distances_batch.view(batch_size, self.num_group, self.num_group).transpose(
            1, 2).contiguous().view(batch_size * self.num_group, self.num_group)
        distances_batch[idx_base] = float("inf")
        distances_batch = distances_batch.view(
            batch_size, self.num_group, self.num_group).transpose(1, 2).contiguous()
        for i in range(self.num_group - 1):
            distances_batch = distances_batch.view(
                batch_size * self.num_group, self.num_group)
            distances_to_last_batch = distances_batch[sorted_indices_list[-1]]
            closest_point_idx = torch.argmin(distances_to_last_batch, dim=-1)
            closest_point_idx = closest_point_idx + idx_base
            sorted_indices_list.append(closest_point_idx)
            distances_batch = distances_batch.view(batch_size, self.num_group, self.num_group).transpose(
                1, 2).contiguous().view(batch_size * self.num_group, self.num_group)
            distances_batch[closest_point_idx] = float("inf")
            distances_batch = distances_batch.view(
                batch_size, self.num_group, self.num_group).transpose(1, 2).contiguous()
        sorted_indices = torch.stack(sorted_indices_list, dim=-1)
        sorted_indices = sorted_indices.view(-1)
        return sorted_indices

    def morton_sorting(self, xyz, center):
        batch_size, num_points, _ = xyz.shape
        all_indices = []
        for index in range(batch_size):
            points = center[index]
            z = get_z_values(points.cpu().numpy())
            idxs = np.zeros((self.num_group), dtype=np.int32)
            temp = np.arange(self.num_group)
            z_ind = np.argsort(z[temp])
            idxs = temp[z_ind]
            all_indices.append(idxs)
        all_indices = torch.tensor(all_indices, device=xyz.device)

        idx_base = torch.arange(
            0, batch_size, device=xyz.device).view(-1, 1) * self.num_group
        sorted_indices = all_indices + idx_base
        sorted_indices = sorted_indices.view(-1)

    def plot_3d_point_cloud(self, point_cloud):
        """
        Plots a 3D scatter plot of the given point cloud with index labels.

        Parameters:
        point_cloud (numpy.ndarray): A numpy array of shape (N, 3) where N is the number of points.
        """
        # Check if the input is a 2D array with 3 columns
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            raise ValueError("Input must be a 2D array with shape (N, 3)")

        # Extract x, y, z coordinates
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]

        # Create a 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers+lines+text',  # Add text labels to the points
            marker=dict(size=5, color='blue'),
            line=dict(width=2, color='lightblue'),
            text=[str(i) for i in range(len(point_cloud))],  # Labels as point indices
            textposition="top center"  # Position of the labels
        )])

        # Set the layout
        fig.update_layout(
            title='3D Scatter Plot of Points with Index Labels',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )

        # Show the plot
        i = np.random.choice(100,1)[0]
        fig.write_html(f'center_plots/{i}.html')

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(
            0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        # can utilize morton_sorting by choosing morton_sorting function
        sorted_indices = self.simplied_morton_sorting(xyz, center)

        neighborhood = neighborhood.view(
            batch_size * self.num_group, self.group_size, 3)[sorted_indices, :, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, 3).contiguous()
        center = center.view(
            batch_size * self.num_group, 3)[sorted_indices, :]
        center = center.view(
            batch_size, self.num_group, 3).contiguous()

        # for i in range(10):
        #     self.plot_3d_point_cloud(center[i].cpu().numpy())
        return neighborhood, center


# Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PositionEmbeddingCoordsSine(nn.Module):
    """Similar to transformer's position encoding, but generalizes it to
    arbitrary dimensions and continuous coordinates.

    Args:
        n_dim: Number of input dimensions, e.g. 2 for image coordinates.
        d_model: Number of dimensions to encode into
        temperature:
        scale:
    """

    def __init__(self, n_dim: int = 1, d_model: int = 256, temperature=10000, scale=None):
        super().__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Point positions (*, d_in)

        Returns:
            pos_emb (*, d_out)
        """
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t,
                                     2, rounding_mode='trunc') / self.num_pos_feats)

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-
                              1).reshape(*xyz.shape[:-1], -1)

        # Pad unused dimensions with zeros
        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb


class GPT_Transformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.decoder_depth = config.transformer_config.decoder_depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        self.group_size = config.group_size
        print_log(f'[args] {config.transformer_config}', logger='Transformer')

        self.encoder_dims = config.transformer_config.encoder_dims

        assert self.encoder_dims in [256, 384, 768, 1024]
        if self.encoder_dims == 384 or self.encoder_dims == 256:
            self.encoder = Encoder_small(encoder_channel=self.encoder_dims)
        else:
            self.encoder = Encoder_large(encoder_channel=self.encoder_dims)

        self.pos_embed = PositionEmbeddingCoordsSine(3, self.encoder_dims, 1.0)

        self.blocks = GPT_extractor(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.depth,
            num_classes=config.cls_dim,
            trans_dim=self.trans_dim,
            group_size=self.group_size,
            pretrained=True,
        )

        self.generator_blocks = GPT_generator(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.decoder_depth,
            trans_dim=self.trans_dim,
            group_size=self.group_size
        )

        # do not perform additional mask on the first (self.keep_attend) tokens
        self.keep_attend = 10
        self.num_groups = config.num_group
        self.num_mask = int(
            (self.num_groups - self.keep_attend) * self.mask_ratio)

        self.sos_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, neighborhood, center, noaug=False, classify=False):
        # generate mask

        group_input_tokens = self.encoder(neighborhood)  # B G C

        batch_size, seq_len, C = group_input_tokens.size()

        relative_position = center[:, 1:, :] - center[:, :-1, :]
        relative_norm = torch.norm(relative_position, dim=-1, keepdim=True)
        relative_direction = relative_position / relative_norm
        position = torch.cat(
            [center[:, 0, :].unsqueeze(1), relative_direction], dim=1)
        pos_relative = self.pos_embed(position)

        sos_pos = self.sos_pos.expand(batch_size, -1, -1)
        pos_absolute = self.pos_embed(center[:, :-1, :])
        pos_absolute = torch.cat([sos_pos, pos_absolute], dim=1)

        attn_mask = torch.full(
            (seq_len, seq_len), -float("Inf"), device=group_input_tokens.device, dtype=group_input_tokens.dtype
        ).to(torch.bool)

        with torch.no_grad():
            attn_mask = torch.triu(attn_mask, diagonal=1)

            # point wise
            # overall_mask = np.zeros([self.num_groups, self.num_groups])
            # for i in range(self.num_groups):
            #     mask = np.hstack([
            #         np.zeros(self.num_groups-self.num_mask),
            #         np.ones(self.num_mask),
            #     ])
            #     np.random.shuffle(mask)
            #     overall_mask[i, :] = mask
            # overall_mask = torch.from_numpy(
            #     overall_mask).to(torch.bool).to('cuda')

            # column wise
            overall_mask = np.hstack([
                np.zeros(self.num_groups-self.keep_attend-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(overall_mask)
            overall_mask = np.hstack([
                np.zeros(self.keep_attend),
                overall_mask,
            ])
            overall_mask = torch.from_numpy(
                overall_mask).to(torch.bool).to('cuda')

            eye_mask = torch.eye(self.num_groups).to(torch.bool).to('cuda')

            attn_mask = attn_mask | overall_mask.unsqueeze(0) & ~eye_mask

        # transformer
        if classify == False:
            encoded_features = self.blocks(
                group_input_tokens, pos_absolute, attn_mask, classify=classify)
            generated_points = self.generator_blocks(
                encoded_features, pos_relative, attn_mask)
            return generated_points
        else:
            print('----error---- This code is detached ----error----')
            logits, generated_points = self.blocks(
                group_input_tokens, pos_absolute, classify=classify)
            return logits, generated_points


@MODELS.register_module()
class PointGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointGPT] ', logger='PointGPT')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.GPT_Transformer = GPT_Transformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.weight_center = config.weight_center

        print_log(
            f'[PointGPT] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger='PointGPT')
        self.group_divider = Group(
            num_group=self.num_group, group_size=self.group_size)

        self.loss = config.loss

        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func_p = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func_p = ChamferDistanceL2().cuda()
        elif loss_type == 'cdl12':
            self.loss_func_p1 = ChamferDistanceL1().cuda()
            self.loss_func_p2 = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
        self.loss_func_c = nn.MSELoss().cuda()

    def forward(self, pts, vis=False, **kwargs):
        neighborhood, center = self.group_divider(pts)

        B = neighborhood.shape[0]

        generated_points = self.GPT_Transformer(
            neighborhood, center)

        gt_points = neighborhood.reshape(
            B*(self.num_group), self.group_size, 3)
        loss1 = self.loss_func_p1(generated_points, gt_points)
        loss2 = self.loss_func_p2(generated_points, gt_points)

        if vis:  # visualization
            gt_points = gt_points.reshape(
                B, self.num_group, self.group_size, 3)
            gt_points = (gt_points + center.unsqueeze(-2)
                         ).reshape(-1, 3).unsqueeze(0)
            generated_points = generated_points.reshape(
                B, self.num_group, self.group_size, 3) + center.unsqueeze(-2)
            generated_points = generated_points.reshape(-1, 3).unsqueeze(0)

            return generated_points, gt_points, center

        return loss1 + loss2


@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.decoder_depth = config.decoder_depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.cls_method = config.cls_method

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.triplet_loss = config.triplet_loss

        self.group_divider = Group(
            num_group=self.num_group, group_size=self.group_size)

        assert self.encoder_dims in [256, 384, 768, 1024]
        if self.encoder_dims == 384 or self.encoder_dims == 256:
            self.encoder = Encoder_small(encoder_channel=self.encoder_dims)
        else:
            self.encoder = Encoder_large(encoder_channel=self.encoder_dims)

        self.pos_embed = PositionEmbeddingCoordsSine(3, self.encoder_dims, 1.0)

        self.blocks = GPT_extractor(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.depth,
            num_classes=config.cls_dim,
            trans_dim=self.trans_dim,
            group_size=self.group_size
        )

        self.generator_blocks = GPT_generator(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.decoder_depth,
            trans_dim=self.trans_dim,
            group_size=self.group_size
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.sos_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.norm = nn.LayerNorm(self.trans_dim)

        if self.cls_method == 'm1':
            self.cls_encoder = Encoder_small(encoder_channel=self.trans_dim)
        elif self.cls_method == 'm2':
            self.stream_encoder = Encoder_small(encoder_channel=self.cls_dim)
            self.cls_head = nn.Sequential(nn.Linear(self.cls_dim*2, self.cls_dim))
        elif self.cls_method == 'm3':
            self.cls_encoder = PointNetEncoder(encoder_channel=self.trans_dim)
        elif self.cls_method == 'm4':
            self.stream_encoder = PointNetEncoder(encoder_channel=self.trans_dim)
        elif self.cls_method == 'm5':  # using left right and center information before GPT takes input
            self.cls_head_LRC = nn.Sequential(nn.Linear(self.encoder_dims*self.num_group, 3))
        elif self.cls_method == 'm6': # using left right and center information at last step after GPT output
            self.cls_head_LRC = nn.Sequential(nn.Linear(self.cls_dim, 3), )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self, loss_type='cdl12'):
        self.loss_ce = nn.CrossEntropyLoss()
        if loss_type == "cdl1":
            self.loss_func_p = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func_p = ChamferDistanceL2().cuda()
        elif loss_type == 'cdl12':
            self.loss_func_p1 = ChamferDistanceL1().cuda()
            self.loss_func_p2 = ChamferDistanceL2().cuda()
            print('cdl12')
        else:
            raise NotImplementedError
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_LRC = nn.CrossEntropyLoss()
        self.loss_triplet = nn.TripletMarginLoss(margin=23.0, p=2)
        if self.config.frechet_loss:
            self.loss_frechet = FrechetLoss()

    def get_triplet(self, labels, feat):
        anchors = []
        positives = []
        negatives = []
        unique_labels = labels.unique()

        # Ensure labels and features are on the same device (CUDA in this case)
        device = feat.device

        for label in unique_labels:
            pos_indices = (labels == label).nonzero(as_tuple=True)[0]
            neg_indices = (labels != label).nonzero(as_tuple=True)[0]

            # For each positive sample, select a positive and a negative example
            for anchor_idx in pos_indices:
                # Select positive and negative indices
                pos_idx = np.random.choice(pos_indices[pos_indices != anchor_idx].cpu())  # Use CPU for numpy operations
                neg_idx = np.random.choice(neg_indices.cpu())  # Use CPU for numpy operations

                # Convert indices back to torch and move to the same device as features (CUDA)
                pos_idx = torch.tensor(pos_idx).to(device)
                neg_idx = torch.tensor(neg_idx).to(device)

                # Create the triplet (anchor, positive, negative) and ensure they are on CUDA
                anchor = feat[anchor_idx].to(device)
                positive = feat[pos_idx].to(device)
                negative = feat[neg_idx].to(device)

                # Append triplets to lists
                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)

        # Stack the lists into tensors on the correct device
        anchors = torch.stack(anchors).to(device)
        positives = torch.stack(positives).to(device)
        negatives = torch.stack(negatives).to(device)

        return anchors, positives, negatives


    def get_loss_acc(self, ret, gt, class_feat=None, LRC_feat=None, LRC_label=None):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))

        if self.triplet_loss: # Triplet loss if apllied
            anchor, positive, negative = self.get_triplet(gt, class_feat)
            triplet_loss = self.loss_triplet(anchor, positive, negative)
        else:
            triplet_loss = 0
        
        if self.cls_method == 'm5': # classifying left right and center positon
            LRC_loss = self.loss_ce_LRC(LRC_feat, LRC_label.long())
        elif self.cls_method == 'm6':
            LRC_loss = self.loss_ce_LRC(LRC_feat, LRC_label.long())
            LRC_loss = LRC_loss*50
        else:
            LRC_loss = 0
        return loss + triplet_loss + LRC_loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k,
                         v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('GPT_Transformer'):
                    base_ckpt[k[len('GPT_Transformer.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                if 'cls_head_finetune' in k:
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(
                        incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(
                f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    

    def plot_3d_scatter(self,pts):
        """
        Takes a tensor of shape (1, 24, 3) on CUDA, moves it to CPU, and creates a 3D scatter plot.
        
        Parameters:
            pts (torch.Tensor): Input tensor of shape (1, 24, 3) on CUDA.
            
        Returns:
            None
        """
        # Move tensor to CPU and convert to NumPy
        points_np = pts.cpu().numpy().squeeze()  # Shape becomes (24, 3)
        
        # Extract x, y, z coordinates
        x, y, z = points_np[:, 0], points_np[:, 1], points_np[:, 2]
        
        # Create 3D scatter plot using Plotly
        scatter = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=z,  # Color points based on z-axis values
                colorscale='Viridis',
                opacity=0.8
            )
        )

        layout = go.Layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            title='3D Scatter Plot'
        )

        fig = go.Figure(data=[scatter], layout=layout)
        i = np.random.choice(100,1)[0]
        os.makedirs('3d_scatterplots', exist_ok=True)
        fig.write_html(f'3d_scatterplots/{i}.html')
    
    def get_similarity_measure(self, generated, ground_truth):
        B = generated.shape[0]
        frechet_distances = torch.zeros(B)
        
        # Loop through each batch and compute the Fréchet distance
        for i in range(B):
            gen_points = generated[i].cpu().detach().numpy()  # Convert to numpy for similaritymeasures
            gt_points = ground_truth[i].cpu().detach().numpy()  # Convert to numpy for similaritymeasures
            
            frechet_dist = similaritymeasures.frechet_distance(gen_points, gt_points)
            frechet_distances[i] = frechet_dist
        
        return frechet_distances
    
    def forward(self, pts):
        if self.cls_method == 'm1' or self.cls_method == 'm2' or self.cls_method == 'm3' or self.cls_method == 'm4':
            mainS, pts = pts
        
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        # self.plot_3d_scatter(pts[0])
        B, L, _ = group_input_tokens.shape

        if self.cls_method == 'm1':
            cls_tokens = self.cls_encoder(mainS.unsqueeze(1))
        elif self.cls_method == 'm3':
            cls_tokens = self.cls_encoder(mainS).unsqueeze(1)
        elif self.cls_method == 'm4':
            stream_embd = self.stream_encoder(mainS).unsqueeze(1).expand(-1,self.num_group,-1)
            group_input_tokens = group_input_tokens + 0.4*stream_embd
            cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        elif self.cls_method == 'm5': # classifying left right and center positon
            LRC_feat = self.cls_head_LRC(group_input_tokens.reshape(B, -1))
            cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        else:
            cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)

        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)
        sos_pos = self.sos_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = torch.cat([sos_pos, pos], dim=1)

        relative_position = center[:, 1:, :] - center[:, :-1, :]
        relative_norm = torch.norm(relative_position, dim=-1, keepdim=True)
        relative_direction = relative_position / relative_norm
        position = torch.cat(
            [center[:, 0, :].unsqueeze(1), relative_direction], dim=1)
        pos_relative = self.pos_embed(position)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        attn_mask = torch.full(
            (L+2, L+2), -float("Inf"), device=group_input_tokens.device, dtype=group_input_tokens.dtype
        ).to(torch.bool)

        attn_mask = torch.triu(attn_mask, diagonal=1)

        # transformer
        ret, encoded_features, class_feat = self.blocks(x, pos, attn_mask, classify=True)

        if self.cls_method == 'm2':
            stream_ret = self.stream_encoder(mainS.unsqueeze(1)).squeeze(1)
            ret = torch.cat([ret, stream_ret], dim=1)
            ret = self.cls_head(ret)    
        elif self.cls_method == 'm6':
            LRC_feat = self.cls_head_LRC(ret)

        encoded_features = torch.cat(
            [encoded_features[:, 0, :].unsqueeze(1), encoded_features[:, 2:-1, :]], dim=1)

        attn_mask = torch.full(
            (L, L), -float("Inf"), device=group_input_tokens.device, dtype=group_input_tokens.dtype
        ).to(torch.bool)

        attn_mask = torch.triu(attn_mask, diagonal=1)

        generated_points = self.generator_blocks(
            encoded_features, pos_relative, attn_mask)

        neighborhood = neighborhood + center.unsqueeze(2)

        gt_points = neighborhood.reshape(
            B*(self.num_group), self.group_size, 3)

        g = generated_points.reshape(B, -1, 3)
        g_t = gt_points.reshape(B, -1, 3)

        loss1 = self.loss_func_p1(generated_points, gt_points)
        loss2 = self.loss_func_p2(generated_points, gt_points)
        if self.config.frechet_loss:
            loss_frechet = self.loss_frechet(g, g_t)
            print(loss_frechet, self.get_similarity_measure(g, g_t))
        if self.cls_method == 'm6' or self.cls_method == 'm5':
            return ret, loss1 + loss2, class_feat, LRC_feat
        return ret, loss1 + loss2, class_feat
