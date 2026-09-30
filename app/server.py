"""Local web app: upload a CT scan, click a nodule, get the ensemble's malignancy estimate.

    .venv\\Scripts\\python -m app.server            (then open http://127.0.0.1:8000)
    .venv\\Scripts\\python -m app.server --run runs/baseline --port 8000
"""
import argparse
import base64
import io
import tempfile
import threading
import uuid
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pydicom
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import BaseModel

from cadc.predict import load_models, score
from cadc.preprocess import download_series, extract_patch

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
# LIDC-IDRI-0001: one large nodule that all four radiologists rated highly suspicious.
DEMO_SERIES = "1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192"
WINDOWS = {"lung": (-1350, 150), "soft": (-160, 240)}
MAX_SCANS = 4  # uploaded scans kept in memory

app = FastAPI(title="CADC nodule analysis")
state = {"models": [], "device": None, "lidc": {}, "run": None}
scans = OrderedDict()
scans_lock = threading.Lock()


# ---------- loading ----------

def read_series(paths):
    """Read DICOM files (any extension), keep the largest CT series, return (volume, spacing, uid).

    Volume is in HU indexed (row, col, slice), slices sorted from inferior to superior.
    """
    series = {}
    for p in paths:
        try:
            ds = pydicom.dcmread(p, force=True)
            if "PixelData" not in ds or "ImagePositionPatient" not in ds:
                continue
            series.setdefault(str(ds.SeriesInstanceUID), []).append(ds)
        except Exception:
            continue
    if not series:
        raise HTTPException(400, "No readable CT slices found. Upload a .zip of DICOM files or the .dcm files themselves.")
    uid, slices = max(series.items(), key=lambda kv: len(kv[1]))
    if len(slices) < 10:
        raise HTTPException(400, f"Only {len(slices)} slices found; a CT series needs many more.")
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    # Drop duplicate positions, as pylidc does, keeping the first.
    seen, unique = set(), []
    for s in slices:
        z = round(float(s.ImagePositionPatient[2]), 3)
        if z not in seen:
            seen.add(z)
            unique.append(s)
    vol = np.stack(
        [s.pixel_array * float(getattr(s, "RescaleSlope", 1)) + float(getattr(s, "RescaleIntercept", 0))
         for s in unique], axis=-1).astype(np.float32)
    z = [float(s.ImagePositionPatient[2]) for s in unique]
    dz = float(np.median(np.diff(z)))
    row_mm, col_mm = (float(v) for v in unique[0].PixelSpacing)
    return vol, np.array([row_mm, col_mm, dz]), uid


def index_lidc_annotations(patch_dir):
    """Map LIDC series UID -> annotated nodules, from the preprocessed shards (if present)."""
    lidc = {}
    for path in Path(patch_dir).glob("*.npz"):
        try:
            with np.load(path) as z:
                ratings = [r[r > 0].tolist() for r in z["malignancy"]]
                lidc[str(z["series_uid"])] = dict(
                    patient_id=str(z["patient_id"]),
                    nodules=[dict(center=c.tolist(), ratings=r, diameter_mm=float(d))
                             for c, r, d in zip(z["center_ijk"], ratings, z["diameter_mm"])],
                )
        except Exception:
            continue
    return lidc


def register_scan(vol, spacing, uid):
    scan_id = uuid.uuid4().hex[:12]
    with scans_lock:
        scans[scan_id] = dict(vol=vol, spacing=spacing, uid=uid)
        while len(scans) > MAX_SCANS:
            scans.popitem(last=False)
    lidc = state["lidc"].get(uid)
    return dict(
        scan_id=scan_id,
        shape=list(vol.shape),
        spacing_mm=[round(float(s), 3) for s in spacing],
        lidc=None if lidc is None else dict(
            patient_id=lidc["patient_id"],
            nodules=[dict(x=round(n["center"][1]), y=round(n["center"][0]), slice=round(n["center"][2]),
                          ratings=n["ratings"], diameter_mm=round(n["diameter_mm"], 1))
                     for n in lidc["nodules"]],
        ),
    )


def get_scan(scan_id):
    with scans_lock:
        if scan_id not in scans:
            raise HTTPException(404, "Scan expired; please upload it again.")
        scans.move_to_end(scan_id)
        return scans[scan_id]


def to_png(img, lo=-1000, hi=400):
    img = np.clip((img - lo) / (hi - lo), 0, 1) * 255
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


# ---------- API ----------

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    return dict(models=len(state["models"]), device=str(state["device"]), run=state["run"],
                lidc_scans_indexed=len(state["lidc"]))


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, f in enumerate(files):
            data = await f.read()
            if f.filename.lower().endswith(".zip") or data[:4] == b"PK\x03\x04":
                try:
                    with zipfile.ZipFile(io.BytesIO(data)) as zf:
                        for j, name in enumerate(zf.namelist()):
                            if not name.endswith("/"):
                                (tmp / f"z{i}_{j}").write_bytes(zf.read(name))
                except zipfile.BadZipFile:
                    raise HTTPException(400, f"{f.filename} is not a valid zip file.")
            else:
                (tmp / f"f{i}").write_bytes(data)
        vol, spacing, uid = read_series(sorted(tmp.iterdir()))
    return register_scan(vol, spacing, uid)


@app.post("/api/demo")
def demo():
    folder = ROOT / "data" / "demo"
    if not any(folder.glob("*.dcm")):
        try:
            download_series(DEMO_SERIES, folder)
        except Exception as e:
            raise HTTPException(502, f"Could not download the sample scan from TCIA: {e}")
    vol, spacing, uid = read_series(sorted(folder.glob("*.dcm")))
    return register_scan(vol, spacing, uid)


@app.get("/api/scan/{scan_id}/slice/{k}")
def slice_png(scan_id: str, k: int, window: str = "lung"):
    vol = get_scan(scan_id)["vol"]
    k = int(np.clip(k, 0, vol.shape[2] - 1))
    lo, hi = WINDOWS.get(window, WINDOWS["lung"])
    return Response(to_png(vol[:, :, k], lo, hi), media_type="image/png",
                    headers={"Cache-Control": "max-age=3600"})


class Point(BaseModel):
    x: float
    y: float
    slice: float


@app.post("/api/scan/{scan_id}/predict")
def predict(scan_id: str, p: Point):
    if not state["models"]:
        raise HTTPException(503, "No trained models loaded. Train first, or start the app with --run pointing to a run folder.")
    scan = get_scan(scan_id)
    vol = scan["vol"]
    if not (0 <= p.y < vol.shape[0] and 0 <= p.x < vol.shape[1] and 0 <= p.slice < vol.shape[2]):
        raise HTTPException(400, "Point is outside the scan.")
    patch = extract_patch(vol, scan["spacing"], np.array([p.y, p.x, p.slice]))
    mean, per_model = score(state["models"], patch[None], state["device"])
    c = patch.shape[0] // 2
    views = {  # patch is (z, y, x) with z running inferior -> superior; flip so superior is up
        "axial": patch[c],
        "coronal": patch[:, c][::-1],
        "sagittal": patch[:, :, c][::-1],
    }
    prob = float(mean[0])
    return dict(
        probability=prob,
        verdict="Likely malignant" if prob >= 0.5 else "Likely benign",
        per_model=[float(v) for v in per_model[0]],
        views={k: base64.b64encode(to_png(v, -1350, 150)).decode() for k, v in views.items()},
        in_training_data=scan["uid"] in state["lidc"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=ROOT / "runs" / "baseline", help="training run with fold*/ checkpoints")
    parser.add_argument("--which", choices=["final", "best"], default="final")
    parser.add_argument("--patches", type=Path, default=ROOT / "data" / "patches",
                        help="preprocessed shards, used to show LIDC radiologist annotations")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    state["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state["run"] = str(args.run)
    try:
        state["models"] = load_models(args.run, args.which, state["device"])
    except SystemExit as e:
        print(f"WARNING: {e}. The app will open, but cannot analyse nodules until models exist.")
    state["lidc"] = index_lidc_annotations(args.patches)
    print(f"{len(state['models'])} models on {state['device']}, {len(state['lidc'])} LIDC scans indexed")
    print(f"Open http://{args.host}:{args.port} in your browser")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
