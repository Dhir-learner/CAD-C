"""Score nodules with the ensemble of all trained fold models.

Two ways to point it at nodules:

  # Nodules already preprocessed into a shard (e.g. one LIDC patient)
  python -m cadc.predict --run RUN --shard PATCHES/LIDC-IDRI-0001_0012.npz

  # A new CT scan: a folder of DICOM slices plus the nodule centre as it
  # appears in a viewer: column (x), row (y), and slice index counted from
  # the lowest (most inferior) slice, all 0-based.
  python -m cadc.predict --run RUN --dicom /path/to/series --center 312 280 57

Note that LIDC patients were in the training data of 4 of the 5 fold models,
so their scores here are optimistic; use oof_predictions.csv from
cadc.evaluate for honest per-nodule results.
"""
import argparse
from pathlib import Path

import numpy as np
import pydicom
import torch

from cadc.data import NoduleDataset
from cadc.model import ResNet3D
from cadc.preprocess import extract_patch


def load_models(run, which, device):
    paths = sorted(Path(run).glob(f"fold*/{which}.pt"))
    if not paths:
        raise SystemExit(f"no fold*/{which}.pt in {run}")
    models = []
    for p in paths:
        m = ResNet3D().to(device)
        m.load_state_dict(torch.load(p, map_location=device)["model"])
        models.append(m.eval())
    return models


def load_dicom_series(folder):
    """Return (volume in HU indexed (row, col, slice), spacing in mm)."""
    slices = [pydicom.dcmread(p) for p in Path(folder).rglob("*.dcm")]
    if not slices:
        raise SystemExit(f"no .dcm files in {folder}")
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    vol = np.stack([s.pixel_array * float(s.RescaleSlope) + float(s.RescaleIntercept) for s in slices], axis=-1)
    z = [float(s.ImagePositionPatient[2]) for s in slices]
    dz = float(np.median(np.diff(z))) if len(z) > 1 else float(slices[0].SliceThickness)
    row_mm, col_mm = (float(v) for v in slices[0].PixelSpacing)
    return vol.astype(np.float32), np.array([row_mm, col_mm, dz])


@torch.no_grad()
def score(models, patches, device):
    """Mean malignancy probability over fold models, plus each model's probability."""
    ds = NoduleDataset(patches, np.zeros(len(patches)), train=False)
    x = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    per_model = np.stack([torch.sigmoid(m(x)).cpu().numpy() for m in models], axis=1)
    return per_model.mean(axis=1), per_model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path, help="run directory with fold*/ checkpoints")
    parser.add_argument("--which", choices=["final", "best"], default="final", help="checkpoint to use per fold")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--shard", type=Path, help="a .npz shard written by cadc.preprocess")
    src.add_argument("--dicom", type=Path, help="folder with one CT series of .dcm files")
    parser.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "SLICE"),
                        help="nodule centre for --dicom (column, row, slice; 0-based)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(args.run, args.which, device)

    if args.shard:
        z = np.load(args.shard)
        patches = z["patches"]
        if len(patches) == 0:
            raise SystemExit(f"{args.shard.name} has no nodules")
        ratings = [r[r > 0] for r in z["malignancy"]]
        names = [f"nodule {i} (radiologists: {', '.join(map(str, r))}; mean {r.mean():.1f})"
                 for i, r in enumerate(ratings)]
    else:
        if args.center is None:
            parser.error("--dicom needs --center X Y SLICE")
        vol, spacing = load_dicom_series(args.dicom)
        x, y, s = args.center
        patches = extract_patch(vol, spacing, np.array([y, x, s]))[None]
        names = [f"nodule at x={x:g} y={y:g} slice={s:g}"]

    mean, per_model = score(models, patches, device)
    print(f"{len(models)} fold models ({args.which} checkpoints)")
    for name, p, pm in zip(names, mean, per_model):
        verdict = "likely MALIGNANT" if p >= 0.5 else "likely benign"
        print(f"{name}: malignancy probability {p:.3f} -> {verdict} "
              f"[per model: {', '.join(f'{v:.2f}' for v in pm)}]")
    print("Research model trained on radiologist ratings, not biopsy results. Not for clinical use.")


if __name__ == "__main__":
    main()
