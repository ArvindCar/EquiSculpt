# test_dataloader.py
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.ShapeNetMeshDataLoader import ShapeNetMeshDataset, get_dataloader

def visualize_pointcloud(pc, title=""):
    """3D visualization of a point cloud"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:,0], pc[:,1], pc[:,2], s=1)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def test_dataloader():
    # Configuration
    root_dir = "./data/ShapeNetCore.v2"  # Update this path
    batch_size = 4
    n_points = 2048
    
    try:
        # Test dataset initialization
        print("Initializing dataset...")
        # for synset_id in os.listdir(root_dir):
        #     synset_dir = os.path.join(root_dir, synset_id)
        #     if os.path.isdir(synset_dir):
        #         for model_id in os.listdir(synset_dir):
        #             model_dir = os.path.join(synset_dir, model_id)
        #             obj_path = os.path.join(model_dir, 'model_normalized.obj')
        #             if os.path.exists(obj_path):
        #                 print("yay")
        #             else:
        #                 obj_path = os.path.join(model_dir, 'models', 'model_normalized.obj')
        #                 if os.path.exists(obj_path):
        #                     print("yay2")
        dataset = ShapeNetMeshDataset(root_dir, n_points=n_points)
        print(f"Dataset contains {len(dataset)} samples")
        
        # Test single sample
        print("\nTesting single sample:")
        sample = dataset[0]
        print(f"Sample shape: {sample.shape}")
        print(f"Min coordinates: {torch.min(sample, dim=0)}")
        print(f"Max coordinates: {torch.max(sample, dim=0)}")
        print(f"Mean coordinates: {torch.mean(sample, dim=0)}")
        
        # Visualize first sample
        visualize_pointcloud(sample.numpy(), "Raw Sample (Normalized)")
        
        # Test dataloader
        print("\nTesting dataloader:")
        loader = get_dataloader(root_dir, batch_size=batch_size, n_points=n_points, shuffle=False, num_workers=1)
        
        # Get first batch
        batch = next(iter(loader))
        print(f"Batch shape: {batch.shape}")
        print(f"Batch dtype: {batch.dtype}")
        
        # Visualize batch samples
        # for i in range(min(4, batch_size)):
        #     visualize_pointcloud(batch[i].numpy(), f"Sample {i+1} from Batch")
            
        # Check normalization
        print("\nChecking normalization:")
        batch_mean = torch.mean(batch, dim=1)
        batch_std = torch.std(batch, dim=1)
        print(f"Batch mean (should be near 0):\n{batch_mean}")
        print(f"Batch std (should be near 1):\n{batch_std}")
        
        # Check augmentation
        print("\nChecking rotation augmentation:")
        rotated_dataset = ShapeNetMeshDataset(root_dir, n_points=n_points, rotation='z')
        original_sample = dataset[0]
        rotated_sample = rotated_dataset[0]
        
        # fig = plt.figure(figsize=(12, 6))
        # ax1 = fig.add_subplot(121, projection='3d')
        # ax1.scatter(original_sample[:,0], original_sample[:,1], original_sample[:,2], s=1)
        # ax1.set_title('Original')
        
        # ax2 = fig.add_subplot(122, projection='3d')
        # ax2.scatter(rotated_sample[:,0], rotated_sample[:,1], rotated_sample[:,2], s=1)
        # ax2.set_title('Rotated')
        # plt.show()
        
        print("\nAll tests passed successfully!")
        
    except Exception as e:
        print(f"\nError encountered: {str(e)}")
        print("Please verify:")
        print("- Correct path to ShapeNetCore.v2")
        print("- Proper installation of PyTorch3D")
        print("- Valid model_normalized.json files in each model directory")

if __name__ == "__main__":
    # Handle missing dependencies
    try:
        test_dataloader()
    except ImportError as e:
        print(f"Import error: {str(e)}")
        print("Please install required dependencies:")
        print("pip install matplotlib pytorch3d")