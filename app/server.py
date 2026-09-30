"""Local web app: learn about lung cancer, see how the model performs, and analyse a nodule
in an uploaded CT scan with the 5-model ensemble (with a Grad-CAM heatmap of where it looked).

    .venv\\Scripts\\python -m app.server            (then open http://127.0.0.1:8000)
    .venv\\Scripts\\python -m app.server --run runs/baseline --port 8000
"""
import argparse
import base64
import io
import json
import socket
import tempfile
import threading
import time
import uuid
import webbrowser
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from sklearn.metrics import roc_curve

from cadc.data import NoduleDataset
from cadc.predict import load_models, score
from cadc.preprocess import download_series, extract_patch

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
# LIDC-IDRI-0001: one large nodule that all four radiologists rated highly suspicious.
DEMO_SERIES = "1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192"
WINDOWS = {"lung": (-1350, 150), "soft": (-160, 240)}
MAX_SCANS = 4  # uploaded scans kept in memory

PAGES = {"": "index.html", "learn": "learn.html", "analyze": "analyze.html", "model": "model.html", "about": "about.html"}
HEAT_RGB = np.array([255, 138, 61], dtype=np.float32)  # heatmap overlay colour (warm orange)

app = FastAPI(title="CADC nodule analysis")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
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
    return png_bytes(img.astype(np.uint8))


def png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def gradcam(models, patch, device):
    """Grad-CAM over the last residual stage, averaged across models, as a (64, 64, 64) map in [0, 1].

    Shows which parts of the cube pushed the models towards 'malignant'.
    """
    ds = NoduleDataset(patch[None], np.zeros(1), train=False)
    x = ds[0][0][None].to(device)
    crop = ds.crop
    cams = []
    for m in models:
        feats = {}
        hook = m.layers.register_forward_hook(lambda _m, _i, out: feats.__setitem__("a", out))
        try:
            with torch.enable_grad():
                logit = m(x)
                grad = torch.autograd.grad(logit.sum(), feats["a"])[0]
        finally:
            hook.remove()
        a = feats["a"].detach()
        cam = torch.relu((grad.mean(dim=(2, 3, 4), keepdim=True) * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(crop,) * 3, mode="trilinear", align_corners=False)[0, 0]
        cams.append(cam / (cam.max() + 1e-8))
    cam = torch.stack(cams).mean(0).cpu().numpy()
    full = np.zeros(patch.shape, dtype=np.float32)
    s = (patch.shape[0] - crop) // 2
    full[s:s + crop, s:s + crop, s:s + crop] = cam
    return full / (full.max() + 1e-8)


def overlay_png(img, heat, lo=-1350, hi=150):
    """Grey CT slice with the heatmap blended on top in a single warm hue."""
    base = np.clip((img - lo) / (hi - lo), 0, 1)[..., None] * 255
    alpha = np.clip((heat - 0.25) / 0.75, 0, 1)[..., None] * 0.7
    rgb = base * (1 - alpha) + HEAT_RGB * alpha
    return png_bytes(rgb.astype(np.uint8))


def patch_views(patch):
    """Axial, coronal and sagittal centre slices of a (z, y, x) cube; superior is up."""
    c = patch.shape[0] // 2
    return {"axial": patch[c], "coronal": patch[:, c][::-1], "sagittal": patch[:, :, c][::-1]}


# ---------- API ----------

def page(name):
    return lambda: FileResponse(STATIC / PAGES[name])


for _name in PAGES:
    app.add_api_route(f"/{_name}", page(_name), methods=["GET"], include_in_schema=False)


@app.get("/api/metrics")
def metrics():
    """Everything the Model page charts: headline metrics, ROC curves, learning curves."""
    run = Path(state["run"])
    if not (run / "metrics.json").exists():
        raise HTTPException(404, "No evaluation results yet: run cadc.evaluate on the training run.")
    out = json.loads((run / "metrics.json").read_text())

    oof = pd.read_csv(run / "oof_predictions.csv")
    patients = oof.groupby("patient_id").agg(label=("label", "max"), prob=("prob", "max"))
    out["roc"] = {}
    for name, df in [("nodule", oof), ("patient", patients)]:
        fpr, tpr, thr = roc_curve(df.label, df.prob)
        keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 200)).astype(int))
        out["roc"][name] = [[round(float(fpr[i]), 4), round(float(tpr[i]), 4), round(float(min(thr[i], 1)), 4)]
                            for i in keep]

    out["fold_final"] = []
    for fold, g in oof.groupby("fold_dir"):
        pred, y = g.prob.to_numpy() >= 0.5, g.label.to_numpy() == 1
        out["fold_final"].append(dict(fold=fold, n=int(len(g)), auc=out["per_fold_auc"][fold],
                                      accuracy=float((pred == y).mean()), sensitivity=float(pred[y].mean()),
                                      specificity=float((~pred[~y]).mean())))

    hist = [pd.read_csv(f) for f in sorted(run.glob("fold*/history.csv"))]
    n = min(len(h) for h in hist)
    out["history"] = {
        "epochs": list(range(1, n + 1)),
        "auc_folds": [h.auc[:n].round(4).tolist() for h in hist],
        "loss_folds": [h.train_loss[:n].round(4).tolist() for h in hist],
    }
    out["history"]["auc_mean"] = np.mean(out["history"]["auc_folds"], axis=0).round(4).tolist()
    out["history"]["loss_mean"] = np.mean(out["history"]["loss_folds"], axis=0).round(4).tolist()
    out["dataset"] = dict(nodules=int(len(oof)), malignant=int(oof.label.sum()),
                          patients=int(oof.patient_id.nunique()), scans=len(state["lidc"]) or 1018)
    return out


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
    heat = gradcam(state["models"], patch, state["device"])
    views, heats = patch_views(patch), patch_views(heat)
    prob = float(mean[0])
    b64 = lambda data: base64.b64encode(data).decode()
    return dict(
        probability=prob,
        verdict="Likely malignant" if prob >= 0.5 else "Likely benign",
        per_model=[float(v) for v in per_model[0]],
        views={k: b64(to_png(v, -1350, 150)) for k, v in views.items()},
        heatmaps={k: b64(overlay_png(views[k], heats[k])) for k in views},
        in_training_data=scan["uid"] in state["lidc"],
    )


def open_when_ready(host, port, url, timeout=60):
    """Open the browser only once the server accepts connections, so the page loads completely."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.3)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=ROOT / "runs" / "baseline", help="training run with fold*/ checkpoints")
    parser.add_argument("--which", choices=["final", "best"], default="final")
    parser.add_argument("--patches", type=Path, default=ROOT / "data" / "patches",
                        help="preprocessed shards, used to show LIDC radiologist annotations")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = parser.parse_args()

    state["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state["run"] = str(args.run)
    try:
        state["models"] = load_models(args.run, args.which, state["device"])
    except SystemExit as e:
        print(f"WARNING: {e}. The app will open, but cannot analyse nodules until models exist.")
    state["lidc"] = index_lidc_annotations(args.patches)
    print(f"{len(state['models'])} models on {state['device']}, {len(state['lidc'])} LIDC scans indexed")
    url = f"http://{args.host}:{args.port}"
    print(f"Open {url} in your browser (keep this window open while using the app)")
    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(args.host, args.port, url), daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
