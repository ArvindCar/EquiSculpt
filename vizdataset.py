import torch
import open3d as o3d
import os
import numpy as np
from models.vn_pointnet_ae import PointNetAutoencoder
from data_utils.ShapeNetMeshDataLoader import get_dataloader
import argparse
from pathlib import Path
# Argument parser to match model settings
def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', default='PointNetAutoencoder', help='Model name')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size [default: 1]')
    parser.add_argument('--npoint', type=int, default=2048, help='Point Number [default: 2048]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--log_dir', type=str, default='vn_autoencoder', help='Log dir [default: vn_autoencoder]')
    parser.add_argument('--data_root', type=str, default="./data/ShapeNetCore.v2", help='Path to ShapeNetCore.v2 directory')
    parser.add_argument('--pooling', type=str, default='mean',
                        help='Pooling method [default: mean]',
                        choices=['mean', 'max'])
    parser.add_argument('--n_knn', default=20, type=int, 
                       help='Number of nearest neighbors [default: 20]')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of data loading workers [default: 4]')
    return parser.parse_args()

# Load model
def load_model(checkpoints_dir, args):
    model = PointNetAutoencoder(args, channel=3).cuda()
    checkpoint_path = checkpoints_dir / 'best_model.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Visualize point clouds using Open3D
def visualize_pointcloud(original, reconstructed):
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(original)
    print(f"Reconstructed shape: {reconstructed.shape}")
    reconstructed_pcd = o3d.geometry.PointCloud()
    reconstructed_pcd.points = o3d.utility.Vector3dVector(reconstructed)

    original_pcd.paint_uniform_color([0, 0, 1])  # Blue for original
    reconstructed_pcd.paint_uniform_color([1, 0, 0])  # Red for reconstructed

    o3d.visualization.draw_geometries([reconstructed_pcd, original_pcd])

def main():
    args = parse_args()

    # Set device (optional, since no model is used)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load dataset
    val_loader = get_dataloader(
        root_dir=args.data_root,
        batch_size=1,
        n_points=args.npoint,
        split='val',
        rotation='aligned',
        num_workers=1,
        shuffle=False
    )

    point_clouds = []  # List to store multiple point clouds

    # Collect multiple original point clouds
    for i, batch in enumerate(val_loader):
        points = batch.transpose(1, 2).squeeze(0).numpy().T  # Shape (2048, 3)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Assign different colors for each point cloud
        color = np.random.rand(3)  # Random color
        pcd.paint_uniform_color(color)

        point_clouds.append(pcd)

        if i == 4:  # Stop after adding 5 point clouds
            break

    # Create a sphere of radius 1m centered at (0, 0, 0)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([1, 0, 0])  # Red sphere

    # Show all collected point clouds and sphere in the same window
    o3d.visualization.draw_geometries(point_clouds + [sphere])

if __name__ == '__main__':
    main()


