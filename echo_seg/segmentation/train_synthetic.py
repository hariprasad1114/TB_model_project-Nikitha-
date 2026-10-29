#!/usr/bin/env python3
"""Train a compact U-Net for LV segmentation on procedural echo frames.

This makes the segmentation path (and the dashboard Echo view) runnable without
credentialed EchoNet downloads.  For real data use `finetune.py` (ViT encoder).

Example:
    python echo_seg/segmentation/train_synthetic.py \
        --output_dir experiments/echo_seg_demo \
        --epochs 12 --img_size 96 --train_n 192 --val_n 48
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # echo_seg/

from utils.synthetic_echo import SyntheticEchoDataset  # noqa: E402
from segmentation.unet_model import UNet, BCEDiceLoss, dice_coef  # noqa: E402


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_dice = 0.0
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, masks)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += float(loss.item())
        total_dice += dice_coef(logits.detach(), masks)
        n += 1
    return total_loss / max(n, 1), total_dice / max(n, 1)


def save_preview(model, ds, device, path, n=4):
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 8))
    with torch.no_grad():
        for i in range(n):
            sample = ds[i]
            img = sample["image"].unsqueeze(0).to(device)
            mask = sample["mask"][0].numpy()
            prob = torch.sigmoid(model(img))[0, 0].cpu().numpy()
            frame = (sample["image"].permute(1, 2, 0).numpy() * 0.5 + 0.5)
            frame = np.clip(frame, 0, 1)
            axes[0, i].imshow(frame, cmap="gray")
            axes[0, i].set_title("frame")
            axes[1, i].imshow(mask, cmap="gray")
            axes[1, i].set_title("GT mask")
            axes[2, i].imshow(frame, cmap="gray")
            axes[2, i].imshow(prob, cmap="jet", alpha=0.45)
            axes[2, i].set_title("pred overlay")
            for ax in axes[:, i]:
                ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Train UNet on synthetic echo frames")
    ap.add_argument("--output_dir", default="experiments/echo_seg_demo")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=96)
    ap.add_argument("--train_n", type=int, default=192)
    ap.add_argument("--val_n", type=int, default=48)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Leave a core free so the dashboard stays responsive on a laptop.
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SyntheticEchoDataset(args.train_n, args.img_size, seed=args.seed)
    val_ds = SyntheticEchoDataset(args.val_n, args.img_size, seed=args.seed + 9001)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    model = UNet(in_ch=3, out_ch=1, base=args.base).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=1e-5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params:,} train={len(train_ds)} val={len(val_ds)}")

    history = {"train_loss": [], "train_dice": [], "val_loss": [], "val_dice": []}
    best_dice = -1.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_dice = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_dice = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_dice"].append(tr_dice)
        history["val_loss"].append(va_loss)
        history["val_dice"].append(va_dice)
        print(f"epoch {epoch:02d}/{args.epochs}  "
              f"train loss={tr_loss:.4f} dice={tr_dice:.4f} | "
              f"val loss={va_loss:.4f} dice={va_dice:.4f} "
              f"({time.time() - t0:.0f}s)")
        if va_dice > best_dice:
            best_dice = va_dice
            torch.save({
                "model": model.state_dict(),
                "base": args.base,
                "img_size": args.img_size,
                "normalize": True,
                "val_dice": va_dice,
                "epoch": epoch,
            }, os.path.join(args.output_dir, "best_model.pth"))
            print(f"  new best -> {best_dice:.4f}")

    # Final save too (best is what the dashboard uses).
    torch.save({
        "model": model.state_dict(),
        "base": args.base,
        "img_size": args.img_size,
        "normalize": True,
        "val_dice": history["val_dice"][-1],
        "epoch": args.epochs,
    }, os.path.join(args.output_dir, "final_model.pth"))

    results = {
        "model": "UNet (synthetic LV demo)",
        "params": n_params,
        "img_size": args.img_size,
        "epochs": args.epochs,
        "best_val_dice": best_dice,
        "history": history,
        "elapsed_sec": round(time.time() - t0, 1),
        "note": "Trained on procedural synthetic echo frames; not a clinical model.",
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(history["train_dice"], label="train")
    axes[1].plot(history["val_dice"], label="val")
    axes[1].set_title("Dice")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"), dpi=130)
    plt.close()

    save_preview(model, val_ds, device, os.path.join(args.output_dir, "samples.png"))
    print(f"best val Dice={best_dice:.4f}; saved to {args.output_dir}")


if __name__ == "__main__":
    main()
