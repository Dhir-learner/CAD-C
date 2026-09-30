"""Download LIDC-IDRI scans from TCIA one at a time and save 3D nodule patches.

Each scan becomes one shard, <out>/<patient_id>_<scan_id>.npz, holding a
64 mm cube at 1 mm spacing around every annotated nodule plus the
radiologists' malignancy ratings. The raw DICOM is deleted after each scan, so
local disk use stays at a few scans. Scans whose shard already exists are
skipped, so rerunning after a Colab disconnect picks up where it stopped.

    python -m cadc.preprocess --out /content/drive/MyDrive/CADC/patches --work /content/dicom
"""
import argparse
import builtins
import io
import os
import shutil
import sys
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from scipy import ndimage

# pylidc still uses NumPy aliases that were removed in NumPy 1.24.
for _name in ("int", "float", "bool"):
    if not hasattr(np, _name):
        setattr(np, _name, getattr(builtins, _name))

PATCH = 64  # patch edge in voxels, at 1 mm isotropic spacing
HU_MIN, HU_MAX = -1000, 400
MAX_READERS = 4
NBIA_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"


def configure_pylidc(dicom_root):
    # pylidc reads the DICOM root from a config file in the home directory.
    name = "pylidc.conf" if sys.platform.startswith("win") else ".pylidcrc"
    Path.home().joinpath(name).write_text(f"[dicom]\npath = {dicom_root}\nwarn = True\n")


def download_series(series_uid, dest, retries=3):
    """Download one DICOM series from the public NBIA API into dest."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(NBIA_URL, params={"SeriesInstanceUID": series_uid}, timeout=600)
            resp.raise_for_status()
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                # Keep only DICOM: pylidc treats every .dcm file in the folder as a slice.
                for member in zf.namelist():
                    if member.lower().endswith(".dcm"):
                        (dest / Path(member).name).write_bytes(zf.read(member))
            if not any(dest.glob("*.dcm")):
                raise RuntimeError(f"no DICOM files in download of {series_uid}")
            return
        except Exception:
            shutil.rmtree(dest, ignore_errors=True)
            if attempt == retries:
                raise
            time.sleep(10 * attempt)


def load_volume_hu(scan):
    """Return the scan in Hounsfield units, indexed (row, col, slice) like pylidc."""
    images = scan.load_all_dicom_images(verbose=False)
    return np.stack(
        [img.pixel_array * float(img.RescaleSlope) + float(img.RescaleIntercept) for img in images],
        axis=-1,
    ).astype(np.float32)


def extract_patch(vol, spacing, center):
    """Crop a cube around center (voxel coords), resample to 1 mm, return (z, y, x) int16."""
    half = np.ceil(PATCH / 2 / spacing).astype(int) + 1
    c = np.round(center).astype(int)
    lo, hi = c - half, c + half
    shape = np.array(vol.shape)
    crop = vol[tuple(slice(a, b) for a, b in zip(np.maximum(lo, 0), np.minimum(hi, shape)))]
    pad = [(max(0, -a), max(0, b - n)) for a, b, n in zip(lo, hi, shape)]
    crop = np.pad(crop, pad, constant_values=HU_MIN)
    crop = ndimage.zoom(crop, spacing, order=1)
    start = (np.array(crop.shape) - PATCH) // 2
    crop = crop[tuple(slice(s, s + PATCH) for s in start)]
    crop = np.clip(crop, HU_MIN, HU_MAX).astype(np.int16)
    return crop.transpose(2, 0, 1)


def process_scan(scan):
    vol = load_volume_hu(scan)
    spacing = np.array([scan.pixel_spacing, scan.pixel_spacing, scan.slice_spacing], dtype=float)
    patches, ratings, diameters, centers = [], [], [], []
    for anns in scan.cluster_annotations(verbose=False):
        center = np.mean([a.centroid for a in anns], axis=0)
        patches.append(extract_patch(vol, spacing, center))
        r = [a.malignancy for a in anns][:MAX_READERS]
        ratings.append(r + [0] * (MAX_READERS - len(r)))
        diameters.append(np.mean([a.diameter for a in anns]))
        centers.append(center)
    return dict(
        patches=np.array(patches, dtype=np.int16).reshape(-1, PATCH, PATCH, PATCH),
        malignancy=np.array(ratings, dtype=np.int8).reshape(-1, MAX_READERS),
        diameter_mm=np.array(diameters, dtype=np.float32),
        center_ijk=np.array(centers, dtype=np.float32).reshape(-1, 3),
        spacing=spacing.astype(np.float32),
        patient_id=np.array(scan.patient_id),
        series_uid=np.array(scan.series_instance_uid),
    )


def save_shard(path, arrays):
    # Write then rename, so an interrupted save never leaves a shard that looks finished.
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path, help="directory for the .npz shards")
    parser.add_argument("--work", required=True, type=Path, help="scratch directory for raw DICOM")
    parser.add_argument("--limit", type=int, default=0, help="process at most this many scans (0 = all)")
    parser.add_argument("--prefetch", type=int, default=3, help="scans to download ahead of processing")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    configure_pylidc(args.work.resolve())
    import pylidc as pl

    def shard_path(scan):
        return args.out / f"{scan.patient_id}_{scan.id:04d}.npz"

    scans = pl.query(pl.Scan).order_by(pl.Scan.patient_id, pl.Scan.id).all()
    todo = [s for s in scans if not shard_path(s).exists()]
    done = len(scans) - len(todo)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(scans)} scans, {done} already done, processing {len(todo)}", flush=True)

    def series_dir(scan):
        # pylidc looks here first: <root>/<patient>/<study uid>/<series uid>
        return args.work / scan.patient_id / scan.study_instance_uid / scan.series_instance_uid

    # Downloads run in threads; pylidc's database session stays on the main thread.
    pool = ThreadPoolExecutor(max(1, args.prefetch))
    futures = {}

    def submit(i):
        if i < len(todo) and i not in futures:
            futures[i] = pool.submit(download_series, todo[i].series_instance_uid, series_dir(todo[i]))

    failures = 0
    for i, scan in enumerate(todo):
        for j in range(i, i + max(1, args.prefetch)):
            submit(j)
        t0 = time.time()
        try:
            futures.pop(i).result()
            arrays = process_scan(scan)
            save_shard(shard_path(scan), arrays)
            print(f"[{i + 1}/{len(todo)}] {shard_path(scan).name}: "
                  f"{len(arrays['patches'])} nodules, {time.time() - t0:.0f}s", flush=True)
        except Exception:
            failures += 1
            msg = f"{scan.patient_id} scan {scan.id} ({scan.series_instance_uid})\n{traceback.format_exc()}\n"
            with open(args.out / "failed.log", "a") as f:
                f.write(msg)
            print(f"[{i + 1}/{len(todo)}] FAILED {msg}", flush=True)
        finally:
            shutil.rmtree(series_dir(scan), ignore_errors=True)
    pool.shutdown()
    print(f"done, {failures} failures" + (f" (see {args.out / 'failed.log'})" if failures else ""), flush=True)


if __name__ == "__main__":
    main()
