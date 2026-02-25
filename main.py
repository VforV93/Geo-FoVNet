"""
FoV-Net: Rotation-Invariant CAD B-rep Learning via Field-of-View Ray Casting

This is the main training and evaluation script for FOVNet models.

Usage:
    # Train on SolidLetters dataset (trains then tests)
    python modeling.py --mode train --dataset solidletters --graph_path graphs
    
    # Test only with a checkpoint
    python modeling.py --mode test --dataset solidletters --ckpt checkpoints/best.ckpt
    
    # Train with custom vision grid resolution
    python modeling.py --mode train --dataset fusion360 --az 24 --el 12
    
    # Train with specific features disabled
    python modeling.py --mode train --dataset mfcad++ --no_uv --no_face_feat
    
    # Learning curve experiment with 500 training samples
    python modeling.py --mode train --dataset solidletters --lc 500
"""

import argparse
import importlib
import os
import random
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from fovnet.FOVNet import FOVNetModule
from uvnet.UVNet import UVNet

class Config:
    """Configuration container (populated from argparse)."""
    pass


def setup_environment(seed):
    """Configure environment for deterministic training."""
    # Configure CUDA for deterministic behavior
    os.environ["CUDA_CACHE_DISABLE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

    # Configure PyTorch for reproducibility
    torch.set_float32_matmul_precision('medium')
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(seed, workers=True)


def seed_worker(worker_id):
    """Seed DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataset_class_and_paths(config):
    """Import dataset class and configure data paths."""
    dataset_configs = {
        "solidletters": ("datasets.solidletters", "SolidLetters", "data/solidletters"),
        "traceparts": ("datasets.traceparts", "TraceParts", "data/traceparts"),
        "fusion360": ("datasets.fusion360", "Fusion360", "data/fusion360"),
        "mfcad++": ("datasets.mfcad", "MFCAD", "data/mfcad++"),
        "bendfm": ("datasets.bendfm", "BenDFM", "../bendfm/data/bendfm"),
        "wuyts": ("datasets.wuyts", "Wuyts", "../data/wuyts/bend_step_clean"),
    }
    
    dataset_key = config.DATASET.lower()
    if dataset_key not in dataset_configs:
        raise ValueError(f"Unknown dataset: {config.DATASET}. "
                        f"Available datasets: {list(dataset_configs.keys())}")
    
    dataset_module, dataset_class, root = dataset_configs[dataset_key]
    path_train = f"{root}/train/{config.GRAPH_PATH}"
    path_val = f"{root}/val/{config.GRAPH_PATH}"
    path_test = f"{root}/test/{config.GRAPH_PATH}"
    path_test_rotated = f"{root}/test_rotated/{config.GRAPH_PATH}"
    mod = importlib.import_module(dataset_module)
    Dataset = getattr(mod, dataset_class)
    return Dataset, path_train, path_val, path_test, path_test_rotated


def get_dataloaders(config):
    """Create dataloaders with optional subsampling for learning curves."""
    Dataset, path_train, path_val, path_test, path_test_rotated = get_dataset_class_and_paths(config)
    train_dataset = Dataset(
        root_dir=path_train, 
        center_and_scale=False, 
        random_rotate=config.TRAIN_RANDOM_ROTATION
    )
    char2label = train_dataset.char2label

    val_dataset = Dataset(
        root_dir=path_val, 
        center_and_scale=False, 
        random_rotate=False, 
        char2label=char2label
    )
    
    # If NUM_TRAIN_SAMPLES is specified, subsample training set only (validation remains unchanged)
    if config.NUM_TRAIN_SAMPLES is not None:
        # Get all file paths from training dataset
        if hasattr(train_dataset, 'file_paths'):
            all_file_paths = train_dataset.file_paths
        elif hasattr(train_dataset, 'data'):
            # If data is preloaded, we need to get the file paths
            all_file_paths = [Path(train_dataset.data[i]['filename']) for i in range(len(train_dataset.data))]
        else:
            raise RuntimeError("Cannot determine file paths from dataset")
        
        if config.NUM_TRAIN_SAMPLES > len(all_file_paths):
            raise ValueError(
                f"Requested {config.NUM_TRAIN_SAMPLES} training samples, "
                f"but only {len(all_file_paths)} available"
            )
        
        if config.SEGMENTATION:
            print(f"\nLearning curve: {config.NUM_TRAIN_SAMPLES} samples (random)")
            rng = np.random.RandomState(config.SEED)
            indices = np.arange(len(all_file_paths))
            rng.shuffle(indices)
            train_indices = indices[:config.NUM_TRAIN_SAMPLES]
            train_file_paths = [all_file_paths[i] for i in train_indices]
        else:
            # For regression tasks, we don't have classes to stratify on; do random sampling
            if config.REGRESSION:
                print(f"\nLearning curve: {config.NUM_TRAIN_SAMPLES} samples (random - regression)")
                rng = np.random.RandomState(config.SEED)
                indices = np.arange(len(all_file_paths))
                rng.shuffle(indices)
                train_indices = indices[:config.NUM_TRAIN_SAMPLES]
                train_file_paths = [all_file_paths[i] for i in train_indices]
            else:
                print(f"\nLearning curve: {config.NUM_TRAIN_SAMPLES} samples (stratified)")
            if hasattr(train_dataset, 'labels'):
                all_labels = train_dataset.labels
            else:
                all_labels = []
                for fp in all_file_paths:
                    stem = fp.stem
                    if '_' in stem:
                        class_name = stem.split('_')[0].lower()
                    else:
                        class_name = stem[0].lower()
                    all_labels.append(char2label.get(class_name, 0))
            
            train_file_paths, _, train_labels, _ = train_test_split(
                all_file_paths, all_labels, train_size=config.NUM_TRAIN_SAMPLES,
                stratify=all_labels, random_state=config.SEED
            )
            unique_train, counts_train = np.unique(train_labels, return_counts=True)
            print(f"  Class distribution: {dict(zip(unique_train, counts_train))}")
        
        train_dataset = Dataset(
            root_dir=path_train,
            center_and_scale=False,
            random_rotate=config.TRAIN_RANDOM_ROTATION,
            char2label=char2label
        )
        train_dataset.file_paths = train_file_paths

    test_dataset = Dataset(
        root_dir=path_test, 
        center_and_scale=False, 
        random_rotate=False, 
        char2label=char2label
    )
    
    test_dataset_rotated = Dataset(
        root_dir=path_test_rotated, 
        center_and_scale=False, 
        random_rotate=False, 
        char2label=char2label
    )
    
    g = torch.Generator()
    g.manual_seed(config.SEED)
    
    dataloader_kwargs = {
        'num_workers': config.NUM_WORKERS, 'worker_init_fn': seed_worker, 'generator': g,
        'persistent_workers': True, 'pin_memory': True,
        'prefetch_factor': 4 if config.NUM_WORKERS > 0 else None,
    }
    
    collate_fn = train_dataset._collate if config.SEGMENTATION else train_dataset._collate_with_labels
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        **dataloader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        **dataloader_kwargs
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        **dataloader_kwargs
    )
    test_loader_rotated = DataLoader(
        test_dataset_rotated, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        **dataloader_kwargs
    )
    
    return train_loader, val_loader, test_loader, test_loader_rotated, Dataset

def test_model(model, loader, test_set_name, config=None):
    """Test model and print results."""
    model.eval()
    all_preds, all_targets = [], []
    
    pbar = tqdm(loader, desc=f"Testing on {test_set_name}", unit="batch", leave=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            inputs = batch["graph"].to(model.device, non_blocking=True)
            if config.SEGMENTATION:
                labels = inputs.ndata["y"]
            else:
                labels = batch["label"].to(model.device, non_blocking=True)
            
            logits = model(inputs)
            if config.SEGMENTATION:
                preds = torch.argmax(logits, dim=-1)
            elif config.REGRESSION:
                preds = logits.view(-1)
            else:
                preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(labels.detach().cpu().numpy())
            if len(all_preds) > 0:
                if config.REGRESSION:
                    current_mae = mean_absolute_error(np.concatenate(all_targets), np.concatenate(all_preds))
                    metric_name = 'Current MAE'
                    metric_val = f'{current_mae:.4f}'
                else:
                    current_acc = accuracy_score(np.concatenate(all_targets), np.concatenate(all_preds))
                    metric_name = 'Current Acc'
                    metric_val = f'{current_acc:.4f}'
            else:
                metric_name = 'Current Acc'
                metric_val = '0.0000'
            pbar.set_postfix({
                metric_name: metric_val,
                'Batch': f'{batch_idx + 1}/{len(loader)}',
                'Samples': len(np.concatenate(all_targets)) if len(all_targets) > 0 else 0
            })
            del inputs, labels, logits, preds, batch
            

    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_targets = np.concatenate(all_targets) if all_targets else np.array([])

    if all_preds.size == 0:
        print(f"No predictions for {test_set_name}, skipping metrics.")
        return
    if config.REGRESSION:
        mse = mean_squared_error(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        print(f"{test_set_name} MSE: {mse:.4f}, MAE: {mae:.4f}")
    else:
        acc = accuracy_score(all_targets, all_preds)
        print(f"{test_set_name} Accuracy: {acc:.4f}")

    if config.SEGMENTATION:
        num_classes = int(np.max(np.concatenate([all_targets, all_preds])) + 1)
        ious = []
        for cls in range(num_classes):
            pred_mask = (all_preds == cls)
            target_mask = (all_targets == cls)
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            iou = intersection / union if union > 0 else float('nan')
            ious.append(iou)
        mean_iou = np.nanmean(ious)
        print(f"{test_set_name} Mean IoU: {mean_iou:.4f}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Mode: 'train' trains then tests, 'test' only evaluates a checkpoint"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["fusion360", "solidletters", "mfcad++", "traceparts", "bendfm", "wuyts"],
        default="solidletters",
        help="Dataset to use (default: solidletters)"
    )
    parser.add_argument(
        "--graph_path",
        type=str,
        default="graphs",
        help="Path to graph files subdirectory (default: graphs)"
    )
    parser.add_argument(
        "--seed", 
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    # Testing configuration
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to checkpoint file for test mode (required for --mode test)"
    )
    
    # Training configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training and evaluation (default: 64)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Maximum number of training epochs (default: 1000)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early stopping patience in epochs (default: 25)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=12,
        help="Number of data loading workers (default: 12)"
    )
    
    # Data augmentation
    parser.add_argument(
        "--aug", 
        action="store_true",
        help="Apply random 3D rotation during training"
    )
    parser.add_argument(
        "--lc",
        type=int,
        help="Number of training samples for learning curves (uses stratified sampling)"
    )
    
    # Model architecture - Vision features
    parser.add_argument(
        "--no_vision", 
        action="store_true",
        help="Disable all vision features"
    )
    parser.add_argument(
        "--az",
        type=int,
        default=12,
        help="Azimuth resolution for vision grids (default: 12)"
    )
    parser.add_argument(
        "--el",
        type=int,
        default=6,
        help="Elevation resolution for vision grids (default: 6)"
    )
    parser.add_argument(
        "--no_ov", 
        action="store_true",
        help="Disable outer vision (upper hemisphere)"
    )
    parser.add_argument(
        "--no_iv", 
        action="store_true",
        help="Disable inner vision (lower hemisphere)"
    )
    
    # Model architecture - UV and surface features
    parser.add_argument(
        "--no_uv", 
        action="store_true",
        help="Disable UV coordinate features"
    )
    parser.add_argument(
        "--global_uv", 
        action="store_true",
        help="Use global UV coordinates instead of local LRF-UV"
    )
    parser.add_argument(
        "--no_face_feat", 
        action="store_true",
        help="Disable surface type and area features"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log to Weights & Biases"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["fovnet", "uvnet"],
        default="fovnet",
        help="Model architecture to use: fovnet (default) or uvnet"
    )
    
    return parser.parse_args()

def main():
    """Run training or testing pipeline."""
    args = parse_args()
    
    if args.mode == "test" and not args.ckpt:
        print("Error: --ckpt is required when --mode is 'test'")
        sys.exit(1)
    
    if args.mode == "test" and args.ckpt and not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint file not found: {args.ckpt}")
        sys.exit(1)
    
    config = Config()
    config.DATASET = args.dataset
    config.GRAPH_PATH = args.graph_path
    config.SEED = args.seed
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.PATIENCE = args.patience
    config.NUM_WORKERS = args.num_workers
    config.TRAIN_RANDOM_ROTATION = args.aug
    config.NUM_TRAIN_SAMPLES = args.lc
    config.TEST_ONLY = (args.mode == "test")
    config.CKPT_PATH = args.ckpt
    config.VISION = not args.no_vision
    config.USE_OV = not args.no_ov
    config.USE_IV = not args.no_iv
    config.USE_UV = not args.no_uv
    config.LOCAL_UV = not args.global_uv
    config.USE_FACE_FEAT = not args.no_face_feat
    config.VISION_AZ = args.az
    config.VISION_EL = args.el
    config.WANDB = args.wandb
    config.MODEL = args.model
    
    print("\n" + "="*80)
    print(f"{config.MODEL.upper()} Training & Evaluation - Mode: {args.mode.upper()}")
    print("="*80)
    print(f"\nDataset Configuration:")
    print(f"  Dataset: {config.DATASET}")
    print(f"  Graph path: {config.GRAPH_PATH}")
    print(f"  Random seed: {config.SEED}")
    
    if config.TEST_ONLY:
        print(f"Checkpoint: {config.CKPT_PATH}")
    else:
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Learning rate: {config.LEARNING_RATE}")
        print(f"Max epochs: {config.EPOCHS}")
        print(f"Patience: {config.PATIENCE}")
        print(f"Rotation augmentation: {config.TRAIN_RANDOM_ROTATION}")
        if config.NUM_TRAIN_SAMPLES:
            print(f"Training samples (learning curve): {config.NUM_TRAIN_SAMPLES}")
    
    print(f"\nModel Configuration:")
    print(f"  Vision: {config.VISION}")
    if config.VISION:
        print(f"    Grid resolution: {config.VISION_AZ} x {config.VISION_EL} (az x el)")
        print(f"    Outer vision: {config.USE_OV}")
        print(f"    Inner vision: {config.USE_IV}")
    print(f"  UV features: {config.USE_UV}")
    if config.USE_UV:
        print(f"    Local UV (LRF-UV): {config.LOCAL_UV}")
    print(f"  Surface features: {config.USE_FACE_FEAT}")
    print(f"  Wandb logging: {config.WANDB}")
    print(f"  Model: {config.MODEL}")
    print("=" * 80)
    
    setup_environment(config.SEED)
    config.SEGMENTATION = True if config.DATASET.lower() in ["fusion360", "mfcad++"] else False
    config.REGRESSION = True if config.DATASET.lower() in ["wuyts"] else False
    
    train_loader, val_loader, test_loader, test_loader_rotated, Dataset = get_dataloaders(config)

    if config.MODEL.lower() == "fovnet":
        model = FOVNetModule(
            num_classes=Dataset.num_classes(),
            lr=config.LEARNING_RATE,
            vision=config.VISION,
            vision_az=config.VISION_AZ,
            vision_el=config.VISION_EL,
            local_uv=config.LOCAL_UV,
            segmentation=config.SEGMENTATION,
            use_face_feat=config.USE_FACE_FEAT,
            use_uv=config.USE_UV,
            use_ov=config.USE_OV,
            use_iv=config.USE_IV
        )
    elif config.MODEL.lower() == "uvnet":
        model = UVNet(num_classes=Dataset.num_classes(), segmentation=config.SEGMENTATION, lr=config.LEARNING_RATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if config.TEST_ONLY:
        print(f"Loading checkpoint from {config.CKPT_PATH}...")
        if config.MODEL.lower() == "fovnet":
            model = FOVNetModule.load_from_checkpoint(config.CKPT_PATH, weights_only=True)
        else:
            model = UVNet.load_from_checkpoint(config.CKPT_PATH, weights_only=True)
        model.to(device)
        
        test_model(model, test_loader, "test", config=config)
        test_model(model, test_loader_rotated, "test_rotated", config=config)
        return

    run_path = Path("checkpoints") / time.strftime("%m%d") / time.strftime("%H%M")
    run_path.mkdir(parents=True, exist_ok=True)

    loggers = [TensorBoardLogger("checkpoints", name=f"{config.DATASET}_{time.strftime('%m%d')}_{time.strftime('%H%M')}")]
    if config.WANDB:
        loggers.append(WandbLogger(project="raycad", name=f"{config.DATASET}_{time.strftime('%m%d')}_{time.strftime('%H%M')}"))

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss", 
            dirpath=str(run_path), 
            filename="best", 
            save_last=True, 
            save_top_k=1, 
            mode="min"
        ),
        EarlyStopping(
            monitor="val_loss", 
            patience=config.PATIENCE, 
            mode="min", 
            min_delta=0.001
        )
    ]

    trainer = Trainer(
        max_epochs=config.EPOCHS,
        callbacks=callbacks,
        logger=loggers,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=100,
        deterministic="warn",
        enable_progress_bar=True,
        enable_model_summary=True
    )
    trainer.fit(model, train_loader, val_loader)

    best_ckpt_path = run_path / "best.ckpt"
    if os.path.exists(best_ckpt_path):
        print(f"\nLoading best model from {best_ckpt_path} for testing.")
        if config.MODEL.lower() == "fovnet":
            model = FOVNetModule.load_from_checkpoint(best_ckpt_path, weights_only=True)
        else:
            model = UVNet.load_from_checkpoint(best_ckpt_path, weights_only=True)
        model.to(device)
        test_model(model, test_loader, "test", config=config)
        test_model(model, test_loader_rotated, "test_rotated", config=config)
    else:
        print(f"Checkpoint {best_ckpt_path} not found after training. Skipping test.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopping training...")
    finally:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()