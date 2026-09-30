"""Train the nodule malignancy classifier on one cross-validation fold.

Checkpoints are written every epoch to <out>/fold<k>/last.pt, and training
resumes from there automatically, so a Colab disconnect costs at most one
epoch. The best epoch by validation AUC is kept as best.pt, with its
validation predictions in val_predictions.csv; the last epoch is kept as
final.pt, with val_predictions_final.csv.

    python -m cadc.train --data /content/patches --out /content/drive/MyDrive/CADC/runs/baseline --fold 0
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from cadc.data import NoduleDataset, assign_folds, load_nodules
from cadc.model import ResNet3D


def save_atomic(obj, path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_one_epoch(model, loader, loss_fn, opt, scaler, device, amp):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    probs = []
    for x, _ in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            probs.append(torch.sigmoid(model(x.to(device)).float()).cpu())
    return torch.cat(probs).numpy()


def metrics(labels, probs):
    pred = probs >= 0.5
    pos, neg = labels == 1, labels == 0
    return dict(
        auc=float(roc_auc_score(labels, probs)),
        accuracy=float((pred == pos).mean()),
        sensitivity=float(pred[pos].mean()),
        specificity=float((~pred[neg]).mean()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path, help="directory of .npz shards")
    parser.add_argument("--out", required=True, type=Path, help="run directory")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-readers", type=int, default=1, help="drop nodules rated by fewer radiologists")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed + args.fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    out = args.out / f"fold{args.fold}"
    out.mkdir(parents=True, exist_ok=True)

    patches, table = load_nodules(args.data, args.min_readers)
    # The split depends only on the data and seed, so every fold (and every resume) sees the same one.
    table = assign_folds(table, args.folds, args.seed)
    is_val = (table["fold"] == args.fold).to_numpy()
    tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    labels = table["label"].to_numpy()
    print(f"device={device} | nodules={len(table)} ({labels.sum()} malignant) | "
          f"patients={table['patient_id'].nunique()} | train={len(tr)} val={len(va)}", flush=True)

    train_loader = DataLoader(
        NoduleDataset(patches[tr], labels[tr], train=True), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=amp, drop_last=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        NoduleDataset(patches[va], labels[va], train=False), batch_size=args.batch_size * 2,
        num_workers=args.workers, pin_memory=amp, persistent_workers=args.workers > 0,
    )

    model = ResNet3D().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    pos_weight = torch.tensor((labels[tr] == 0).sum() / max(1, labels[tr].sum()), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    start_epoch, best_auc = 0, -1.0
    last = out / "last.pt"
    if last.exists():
        ckpt = torch.load(last, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_auc = ckpt["epoch"] + 1, ckpt["best_auc"]
        print(f"resumed from epoch {ckpt['epoch']} (best AUC {best_auc:.4f})", flush=True)
    (out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    history = out / "history.csv"
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, opt, scaler, device, amp)
        sched.step()
        probs = predict(model, val_loader, device, amp)
        m = metrics(labels[va], probs)
        row = dict(epoch=epoch, train_loss=round(train_loss, 5), lr=opt.param_groups[0]["lr"],
                   **{k: round(v, 5) for k, v in m.items()}, seconds=round(time.time() - t0, 1))

        if m["auc"] > best_auc:
            best_auc = m["auc"]
            save_atomic({"model": model.state_dict(), "epoch": epoch, "metrics": m}, out / "best.pt")
            table.iloc[va].assign(prob=probs).to_csv(out / "val_predictions.csv", index=False)
        if epoch == args.epochs - 1:
            # The final epoch is chosen without looking at validation scores, so these
            # predictions give an unbiased estimate; the best-epoch ones are optimistic.
            save_atomic({"model": model.state_dict(), "epoch": epoch, "metrics": m}, out / "final.pt")
            table.iloc[va].assign(prob=probs).to_csv(out / "val_predictions_final.csv", index=False)
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": epoch, "best_auc": best_auc}, last)

        new_file = not history.exists()
        with open(history, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        print(" | ".join(f"{k}={v}" for k, v in row.items()) + f" | best_auc={best_auc:.4f}", flush=True)

    (out / "summary.json").write_text(json.dumps({"fold": args.fold, "best_auc": best_auc}, indent=2))
    print(f"fold {args.fold} finished, best AUC {best_auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
