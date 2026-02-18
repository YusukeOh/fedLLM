"""
Training script for FedTimeLLM on FRED-MD.

Usage
-----
    python scripts/run_train.py --config configs/default.yaml
    python scripts/run_train.py --config configs/default.yaml --pred_len 6
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.fred_md import FREDMDDataset
from src.evaluation.metrics import compute_all
from src.models.time_llm.model import FedTimeLLM


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str, overrides: dict) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    flat = {}
    flat.update(cfg.get("data", {}))
    flat.update(cfg.get("model", {}))
    flat.update(cfg.get("training", {}))

    for k, v in overrides.items():
        if v is not None:
            flat[k] = v

    return flat


def build_dataloaders(cfg: dict):
    common = dict(
        root_path=cfg["root_path"],
        features=cfg["features"],
        target=cfg["target"],
        scale=cfg["scale"],
        freq=cfg["freq"],
    )
    size = [cfg["seq_len"], cfg["label_len"], cfg["pred_len"]]

    train_ds = FREDMDDataset(flag="train", size=size, **common)
    val_ds = FREDMDDataset(flag="val", size=size, **common)
    test_ds = FREDMDDataset(flag="test", size=size, **common)

    cfg["enc_in"] = train_ds.enc_in

    bs = cfg["batch_size"]
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, drop_last=True)

    return train_ds, val_ds, test_ds, train_dl, val_dl, test_dl


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = []
    total_mae = []
    mae_fn = nn.L1Loss()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in dataloader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -batch_y.shape[1] :, :]).to(device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            pred_len = outputs.shape[1]
            batch_y = batch_y[:, -pred_len:, :]

            loss = criterion(outputs, batch_y)
            mae_loss = mae_fn(outputs, batch_y)
            total_loss.append(loss.item())
            total_mae.append(mae_loss.item())

    return np.mean(total_loss), np.mean(total_mae)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--pred_len", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--llm_model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg = load_config(args.config, overrides)

    seed_everything(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading FRED-MD data ...")
    train_ds, val_ds, test_ds, train_dl, val_dl, test_dl = build_dataloaders(cfg)
    print(f"  Train: {len(train_ds)} samples | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"  Variables: {cfg['enc_in']} | Seq: {cfg['seq_len']} → Pred: {cfg['pred_len']}")

    print("Building model ...")
    model = FedTimeLLM(cfg).float().to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["learning_rate"],
    )

    lr_sched = cfg.get("lr_scheduler", "cosine")
    if lr_sched == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-8)
    else:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg["learning_rate"],
            steps_per_epoch=len(train_dl),
            epochs=cfg["epochs"],
        )

    ckpt_dir = os.path.join("checkpoints", f"fedtimellm_pred{cfg['pred_len']}")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    patience = cfg.get("patience", 10)

    print("\n=== Training ===")
    for epoch in range(cfg["epochs"]):
        model.train()
        losses = []
        t0 = time.time()

        for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(
            train_dl, desc=f"Epoch {epoch + 1}/{cfg['epochs']}", leave=False
        ):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -batch_y.shape[1] :, :]).to(device)

            optimizer.zero_grad()
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            pred_len = outputs.shape[1]
            loss = criterion(outputs, batch_y[:, -pred_len:, :])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if lr_sched == "cosine":
            scheduler.step()

        train_loss = np.mean(losses)
        val_loss, val_mae = evaluate(model, val_dl, criterion, device)
        test_loss, test_mae = evaluate(model, test_dl, criterion, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Train {train_loss:.6f} | Val {val_loss:.6f} | Test {test_loss:.6f} | "
            f"MAE {test_mae:.6f} | {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

    print("\n=== Final evaluation ===")
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best.pt"), weights_only=True))
    test_loss, test_mae = evaluate(model, test_dl, criterion, device)
    print(f"Test MSE: {test_loss:.6f} | Test MAE: {test_mae:.6f}")


if __name__ == "__main__":
    main()
