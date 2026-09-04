"""
FoV-Net: Rotation-Invariant CAD B-rep Learning via Field-of-View Ray Casting

Usage:
    python main.py --mode train --dataset solidletters --graph_path graphs
    python main.py --mode test --dataset solidletters --ckpt checkpoints/best.ckpt
    python main.py --mode train --dataset fusion360 --az 24 --el 12
    python main.py --mode train --dataset mfcad++ --no_uv --no_face_feat
    python main.py --mode train --dataset mfcad++ --lc 500
"""

import argparse
import gc
import importlib
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from fovnet import FOVNetModule

# ── Dataset registry ────────────────────────────────────────────────────────
DATASET_REGISTRY = {
    "solidletters": ("datasets.solidletters", "SolidLetters", "data/solidletters"),
    "traceparts":   ("datasets.traceparts",   "TraceParts",   "data/traceparts"),
    "fusion360":    ("datasets.fusion360",     "Fusion360",    "data/fusion360"),
    "mfcad++":      ("datasets.mfcad",         "MFCAD",        "data/mfcad++"),
}

SEGMENTATION_DATASETS = {"fusion360", "mfcad++"}


# ── Environment ─────────────────────────────────────────────────────────────
def setup_environment(seed: int):
    """Configure deterministic training."""
    os.environ["CUDA_CACHE_DISABLE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

    torch.set_float32_matmul_precision("medium")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(seed, workers=True)


def _seed_worker(_worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


# ── Data loading ────────────────────────────────────────────────────────────
def _load_dataset_class(name: str):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASET_REGISTRY)}")
    mod_name, cls_name, _ = DATASET_REGISTRY[name]
    return getattr(importlib.import_module(mod_name), cls_name)


def _make_paths(dataset: str, graph_path: str):
    root = DATASET_REGISTRY[dataset][2]
    return {s: f"{root}/{s}/{graph_path}" for s in ("train", "val", "test", "test_rotated")}


def get_dataloaders(cfg):
    """Create train / val / test / test_rotated dataloaders."""
    Dataset = _load_dataset_class(cfg.DATASET)
    paths = _make_paths(cfg.DATASET, cfg.GRAPH_PATH)

    train_ds = Dataset(root_dir=paths["train"], random_rotate=cfg.TRAIN_RANDOM_ROTATION)
    char2label = train_ds.char2label
    val_ds = Dataset(root_dir=paths["val"], char2label=char2label)

    if cfg.NUM_TRAIN_SAMPLES is not None:
        train_ds = _subsample_train(train_ds, Dataset, paths["train"], char2label, cfg)

    test_ds = Dataset(root_dir=paths["test"], char2label=char2label)
    test_rot_ds = Dataset(root_dir=paths["test_rotated"], char2label=char2label)

    g = torch.Generator()
    g.manual_seed(cfg.SEED)
    common = dict(
        num_workers=cfg.NUM_WORKERS, worker_init_fn=_seed_worker, generator=g,
        persistent_workers=True, pin_memory=True,
        prefetch_factor=4 if cfg.NUM_WORKERS > 0 else None,
    )
    collate = train_ds._collate if cfg.SEGMENTATION else train_ds._collate_with_labels

    def loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, collate_fn=collate, **common)

    return loader(train_ds, shuffle=True), loader(val_ds), loader(test_ds), loader(test_rot_ds), Dataset


def _subsample_train(train_ds, Dataset, path_train, char2label, cfg):
    """Sub-sample training set for learning-curve experiments."""
    fps = train_ds.file_paths
    n = cfg.NUM_TRAIN_SAMPLES
    if n > len(fps):
        raise ValueError(f"Requested {n} samples but only {len(fps)} available")

    rng = np.random.RandomState(cfg.SEED)

    if cfg.SEGMENTATION or cfg.REGRESSION:
        tag = "random" if cfg.SEGMENTATION else "random - regression"
        print(f"\nLearning curve: {n} samples ({tag})")
        selected = [fps[i] for i in rng.permutation(len(fps))[:n]]
    else:
        print(f"\nLearning curve: {n} samples (stratified)")
        labels = getattr(train_ds, "labels", None)
        if labels is None:
            labels = [char2label.get(p.stem.split("_")[0].lower(), 0) for p in fps]
        selected, _, _, _ = train_test_split(
            fps, labels, train_size=n, stratify=labels, random_state=cfg.SEED
        )

    new_ds = Dataset(root_dir=path_train, random_rotate=cfg.TRAIN_RANDOM_ROTATION, char2label=char2label)
    new_ds.file_paths = selected
    return new_ds


# ── Evaluation ──────────────────────────────────────────────────────────────
def test_model(model, loader, name: str, cfg):
    """Run inference and print metrics."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Testing {name}", unit="batch"):
            inputs = batch["graph"].to(model.device, non_blocking=True)
            labels = inputs.ndata["y"] if cfg.SEGMENTATION else batch["label"].to(model.device, non_blocking=True)
            logits = model(inputs)
            preds = logits.view(-1) if cfg.REGRESSION else logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    preds, targets = np.concatenate(all_preds), np.concatenate(all_targets)
    if preds.size == 0:
        print(f"No predictions for {name}.")
        return

    if cfg.REGRESSION:
        print(f"{name} MSE: {mean_squared_error(targets, preds):.4f}, MAE: {mean_absolute_error(targets, preds):.4f}")
    else:
        print(f"{name} Accuracy: {accuracy_score(targets, preds):.4f}")

    if cfg.SEGMENTATION:
        n_cls = int(max(targets.max(), preds.max()) + 1)
        ious = []
        for c in range(n_cls):
            inter = ((preds == c) & (targets == c)).sum()
            union = ((preds == c) | (targets == c)).sum()
            ious.append(inter / union if union > 0 else float("nan"))
        print(f"{name} Mean IoU: {np.nanmean(ious):.4f}")


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "test"], default="train")
    p.add_argument("--dataset", choices=list(DATASET_REGISTRY), default="solidletters")
    p.add_argument("--graph_path", default="graphs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt", type=str, help="Checkpoint for test mode")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--aug", action="store_true", help="Random 3D rotation augmentation")
    p.add_argument("--lc", type=int, help="Learning-curve sample count")
    p.add_argument("--az", type=int, default=12)
    p.add_argument("--el", type=int, default=6)
    p.add_argument("--no_vision", action="store_true")
    p.add_argument("--no_ov", action="store_true", help="Disable outer vision")
    p.add_argument("--no_iv", action="store_true", help="Disable inner vision")
    p.add_argument("--no_uv", action="store_true")
    p.add_argument("--no_face_feat", action="store_true")
    p.add_argument("--global_uv", action="store_true")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.mode == "test":
        if not args.ckpt:
            sys.exit("Error: --ckpt is required for test mode")
        if not os.path.exists(args.ckpt):
            sys.exit(f"Error: checkpoint not found: {args.ckpt}")

    cfg = argparse.Namespace(
        DATASET=args.dataset, GRAPH_PATH=args.graph_path, SEED=args.seed,
        BATCH_SIZE=args.batch_size, EPOCHS=args.epochs, LEARNING_RATE=args.lr,
        PATIENCE=args.patience, NUM_WORKERS=args.num_workers,
        TRAIN_RANDOM_ROTATION=args.aug, NUM_TRAIN_SAMPLES=args.lc,
        TEST_ONLY=(args.mode == "test"), CKPT_PATH=args.ckpt,
        VISION=not args.no_vision, USE_OV=not args.no_ov, USE_IV=not args.no_iv,
        USE_UV=not args.no_uv, LOCAL_UV=not args.global_uv,
        USE_FACE_FEAT=not args.no_face_feat, VISION_AZ=args.az, VISION_EL=args.el,
        WANDB=args.wandb,
        SEGMENTATION=args.dataset.lower() in SEGMENTATION_DATASETS,
        REGRESSION=False,
    )

    _print_config(cfg)
    setup_environment(cfg.SEED)

    train_loader, val_loader, test_loader, test_rot_loader, Dataset = get_dataloaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FOVNetModule(
        num_classes=Dataset.num_classes(), lr=cfg.LEARNING_RATE,
        vision=cfg.VISION, vision_az=cfg.VISION_AZ, vision_el=cfg.VISION_EL,
        local_uv=cfg.LOCAL_UV, segmentation=cfg.SEGMENTATION,
        use_face_feat=cfg.USE_FACE_FEAT, use_uv=cfg.USE_UV,
        use_ov=cfg.USE_OV, use_iv=cfg.USE_IV,
    ).to(device)

    if cfg.TEST_ONLY:
        model = FOVNetModule.load_from_checkpoint(cfg.CKPT_PATH, weights_only=True).to(device)
        test_model(model, test_loader, "test", cfg)
        test_model(model, test_rot_loader, "test_rotated", cfg)
        return

    # ── Training ────────────────────────────────────────────────────────
    ts = time.strftime("%m%d/%H%M")
    run_path = Path("checkpoints") / ts
    run_path.mkdir(parents=True, exist_ok=True)
    run_name = f"{cfg.DATASET}_{ts.replace('/', '_')}"

    loggers = [TensorBoardLogger("checkpoints", name=run_name)]
    if cfg.WANDB:
        loggers.append(WandbLogger(project="fovnet", name=run_name))

    trainer = Trainer(
        max_epochs=cfg.EPOCHS,
        callbacks=[
            ModelCheckpoint(monitor="val_loss", dirpath=str(run_path), filename="best",
                            save_last=True, save_top_k=1, mode="min"),
            EarlyStopping(monitor="val_loss", patience=cfg.PATIENCE, mode="min", min_delta=0.001),
        ],
        logger=loggers,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, log_every_n_steps=100, deterministic="warn",
    )
    trainer.fit(model, train_loader, val_loader)

    best = run_path / "best.ckpt"
    if best.exists():
        model = FOVNetModule.load_from_checkpoint(str(best), weights_only=True).to(device)
        test_model(model, test_loader, "test", cfg)
        test_model(model, test_rot_loader, "test_rotated", cfg)
    else:
        print(f"Checkpoint {best} not found after training.")


def _print_config(cfg):
    print(f"\n{'='*70}")
    print(f"FoVNet — {cfg.DATASET} — {'TEST' if cfg.TEST_ONLY else 'TRAIN'}")
    print(f"{'='*70}")
    print(f"  seed={cfg.SEED}  batch={cfg.BATCH_SIZE}  lr={cfg.LEARNING_RATE}  epochs={cfg.EPOCHS}")
    print(f"  vision={cfg.VISION} (az={cfg.VISION_AZ} el={cfg.VISION_EL} ov={cfg.USE_OV} iv={cfg.USE_IV})")
    print(f"  uv={cfg.USE_UV} (local={cfg.LOCAL_UV})  face_feat={cfg.USE_FACE_FEAT}")
    if cfg.NUM_TRAIN_SAMPLES:
        print(f"  learning-curve samples: {cfg.NUM_TRAIN_SAMPLES}")
    print(f"{'='*70}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
