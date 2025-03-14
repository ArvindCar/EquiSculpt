"""
Modified Training Loop for VN-PointNet Autoencoder
"""

import argparse
import os
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
from models.vn_pointnet_ae import PointNetAutoencoder, ChamferLoss
from data_utils.ShapeNetMeshDataLoader import ShapeNetPointCloudDataset, get_dataloader
import wandb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', default='PointNetAutoencoder', help='Model name')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size [default: 32]')
    parser.add_argument('--epoch', default=200, type=int, help='Epochs [default: 200]')
    parser.add_argument('--learning_rate', default=0.0005, type=float, help='Learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default='vn_autoencoder', help='Log dir [default: vn_autoencoder]')
    parser.add_argument('--npoint', type=int, default=2048, help='Point Number [default: 2048]')
    parser.add_argument('--normal', action='store_true', default=False, help='Use normals [default: False]')
    parser.add_argument('--rot', type=str, default='aligned', 
                        help='Rotation augmentation [default: aligned]',
                        choices=['aligned', 'z', 'so3'])
    parser.add_argument('--pooling', type=str, default='mean',
                        help='Pooling method [default: mean]',
                        choices=['mean', 'max'])
    parser.add_argument('--n_knn', default=20, type=int, 
                       help='Number of nearest neighbors [default: 20]')
    parser.add_argument('--data_root', type=str, default="./data/ShapeNetCore.v2", 
                       help='Path to ShapeNetCore.v2 directory')
    parser.add_argument('--num_workers', type=int, default=1, 
                       help='Number of data loading workers [default: 1]')
    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
    
    # Initialize environment and directories
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = Path('./log/').joinpath(args.log_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Initialize datasets and dataloaders
    train_loader = get_dataloader(
        root_dir=args.data_root,  # Use actual path from args
        batch_size=args.batch_size,
        n_points=args.npoint,
        split='train',
        rotation=args.rot,
        num_workers=args.num_workers  # Add worker count
    )

    val_loader = get_dataloader(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        n_points=args.npoint,
        split='val',
        rotation='aligned',
        num_workers=args.num_workers,
        shuffle=False  # Explicitly disable shuffling for validation
    )
    # Model setup
    model = PointNetAutoencoder(args, channel=3).cuda()  # Only XYZ coordinates
    criterion = ChamferLoss().cuda()

    # Optimizer setup
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=1e-4
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate*100,
            momentum=0.9,
            weight_decay=1e-4
        )

    best_loss = float('inf')
    
    # Training loop
    for epoch in range(args.epoch):
        log_string(f'Epoch {epoch+1}/{args.epoch}:')
        
        # Training phase
        model.train()
        train_loss = []
        for batch in tqdm(train_loader, total=len(train_loader)):
            if batch is None:  # Skip failed batches
                continue
            points = batch.transpose(1, 2).cuda()  # Convert to (B, 3, N)
            
            optimizer.zero_grad()
            reconstructed, _, _ = model(points)
            print("Shapes of points and reconstructed: ", points.size(), reconstructed.size())
            loss = criterion(reconstructed, points)
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            wandb.log({'Train Loss (Batch)': loss.item()})
            print(f'Loss: {loss.item()}')

        # Validation phase
        model.eval()
        val_loss = []
        with torch.no_grad():
            for batch in val_loader:
                points = batch.transpose(1, 2).cuda()
                reconstructed, _, _ = model(points)
                loss = criterion(reconstructed, points)
                val_loss.append(loss.item())
                wandb.log({'Val Loss (Batch)': loss.item()})
                print(f'Val Loss: {loss.item()}')

        # Logging
        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)
        log_string(f'Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}')
        wandb.log({'Train Loss': avg_train_loss, 'Val Loss': avg_val_loss})
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            savepath = checkpoints_dir / 'best_model.pth'
            log_string(f'Saving best model at {savepath}')
            torch.save({
                'epoch': epoch,
                'test_loss': avg_val_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, savepath)
    torch.save(model.state_dict(), checkpoints_dir / 'final_model.pth')
    
if __name__ == '__main__':
    args = parse_args()
    wandb.init(project='VN_PointNet_AE', name=f'FC Hierarchy', config=vars(args))
    main(args)
    # Initialize wandb
    
