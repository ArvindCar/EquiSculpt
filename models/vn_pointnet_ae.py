import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from models.vn_layers import *
from models.utils.vn_dgcnn_util import get_graph_feature_cross
from models.vn_pointnet import STNkd

class PointNetAutoencoder(nn.Module):
    def __init__(self, args, global_feat=True, feature_transform=True, channel=3):
        super(PointNetAutoencoder, self).__init__()
        self.args = args
        self.n_knn = args.n_knn
        self.channel = channel
        self.feature_transform = feature_transform
        self.num_points = 2048
        self.num_coarse = 64  # Number of coarse points
        self.num_fine = self.num_points // self.num_coarse  # Number of fine points per coarse point
        
        # Original Encoder Components
        self.conv_pos = VNLinearLeakyReLU(3, 64//3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64//3, 64//3, dim=4, negative_slope=0.0)
        if self.feature_transform:
            self.conv2 = VNLinearLeakyReLU(128//3, 128//3, dim=4, negative_slope=0.0)
        else:
            self.conv2 = VNLinearLeakyReLU(64//3, 128//3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinear(128//3, 1024//3)
        self.bn3 = VNBatchNorm(1024//3, dim=4)
        self.std_feature = VNStdFeature(2*1024//3, dim=4, normalize_frame=False, negative_slope=0.0)
        
        if args.pooling == 'max':
            self.pool = VNMaxPool(64//3)
        elif args.pooling == 'mean':
            self.pool = mean_pool
            
        # New Decoder Components
        # Hierarchical FC Decoder
        self.fc1 = nn.Linear(1024//3 * 6, self.num_coarse * 256)  # Predict features for 64 points
        self.fc1_xyz = nn.Linear(1024//3 * 6, self.num_coarse * 3)  # Predict XYZ for 64 points

        self.decconv1 = nn.Conv1d(256, 256, 1)  # Process features for each coarse point
        self.decconv2 = nn.Conv1d(256, self.num_fine * 3, 1)  # Predict local XYZ offsets for fine points
        
        self.global_feat = global_feat
        
        
        if self.feature_transform:
            self.fstn = STNkd(args, d=64//3)

    def forward(self, x):
        B, D, N = x.size()
        print(f"B: {B}, D: {D}, N: {N}")
        
        # Encoder
        x = x.unsqueeze(1)
        print("Unsqueeze size: ", x.size())
        feat = get_graph_feature_cross(x, k=self.n_knn)
        print("Graph feature size: ", feat.size())
        x = self.conv_pos(feat)
        print("Conv_pos size: ", x.size())
        x = self.pool(x)
        print("Pool size: ", x.size())
        
        x = self.conv1(x)
        
        if self.feature_transform:
            x_global = self.fstn(x).unsqueeze(-1).repeat(1,1,1,N)
            x = torch.cat((x, x_global), 1)
            
        pointfeat = x
        x = self.conv2(x)
        x = self.bn3(self.conv3(x))
        print("Conv3 size: ", x.size())
        
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        print("Pre cat size: ", x.size())
        x = torch.cat((x, x_mean), 1)
        print("Post cat size: ", x.size())
        x, trans = self.std_feature(x)
        print("Std feature size: ", x.size())
        x = x.view(B, -1, N)
        print("View size: ", x.size())
        
        # Latent representation
        latent = torch.max(x, -1, keepdim=False)[0]  # (B, 6*1024//3)
        print("Latent size: ", latent.size())
        
        # Decoder
        # Predict coarse points and their features
        pc1_feat = self.fc1(latent)  # [B, 64 * 256]
        pc1_xyz = self.fc1_xyz(latent)  # [B, 64 * 3]

        # Reshape coarse points and features
        pc1_feat = pc1_feat.view(B, self.num_coarse, 256)  # [B, 64, 256]
        pc1_xyz = pc1_xyz.view(B, self.num_coarse, 3)  # [B, 64, 3]

        # Refine coarse features into fine points
        pc2_feat = F.relu(self.decconv1(pc1_feat.transpose(1, 2)))  # [B, 256, 64]
        pc2_xyz = self.decconv2(pc2_feat)  # [B, 64, 32 * 3]

        # Reshape fine points
        pc2_xyz = pc2_xyz.view(B, self.num_coarse, self.num_fine, 3)  # [B, 64, 32, 3]

        # Translate local XYZs to global XYZs
        pc1_xyz_expand = pc1_xyz.unsqueeze(2)  # [B, 64, 1, 3]
        pc2_xyz = pc2_xyz + pc1_xyz_expand  # [B, 64, 32, 3]

        # Reshape to final point cloud
        reconstructed = pc2_xyz.view(B, 3, N)  # [B, 3, 2048]

        return reconstructed, latent, trans
        # latent_reshaped = latent.view(B, -1, 3, 1)  # [B, 6*(1024//3)/3, 3, 1]
        # print("Latent reshaped size: ", latent_reshaped.size())
        # expanded_latent = latent_reshaped.repeat(1, 1, 1, N)  # [B, 6*(1024//3)/3, 3, N]
        # print("Expanded latent size: ", expanded_latent.size())

        # reconstructed = self.decoder(expanded_latent)
        # print("Reconstructed size: ", reconstructed.size())
        # reconstructed = reconstructed.squeeze(1)
        
        # return reconstructed, latent, trans     

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, pred, target):
        # pred/target shape: (B, 3, N)
        pred = pred.transpose(2, 1).contiguous()  # (B, N, 3)
        target = target.transpose(2, 1).contiguous()
        
        # Chamfer Distance calculation
        dist = torch.cdist(pred, target)
        
        min_dist_p_to_t = torch.min(dist, dim=2)[0].mean()
        min_dist_t_to_p = torch.min(dist, dim=1)[0].mean()
        
        return min_dist_p_to_t + min_dist_t_to_p