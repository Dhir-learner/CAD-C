"""External test on the SPIE-AAPM-NCI LUNGx challenge: scans the models never saw, with pathology-confirmed nodules.

For every nodule in the challenge spreadsheets (location + confirmed diagnosis), the scan is downloaded from
TCIA, the nodule is scored at its known location by each trained run and by their ensemble, and (optionally)
the automatic detector is run to see whether it would have found the nodule. Scans are deleted after use.

    python -m cadc.external_eval --runs runs/baseline runs/multiview --names "3D ResNet" "2.5D multi-view" \\
        --calibration data/meta/lungx_calibration.xlsx --test data/meta/lungx_test.xlsx --out runs/lungx --detect

Spreadsheets: TCIA wiki, "SPIE-AAPM-NCI Lung Nodule Classification Challenge Dataset"
(CalibrationSet_NoduleData.xlsx, TestSet_NoduleData_PublicRelease_wTruth.xlsx).
"""
import argparse
import io
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import requests
import torch

from cadc.evaluate import summarize
from cadc.predict import load_models, score
from cadc.preprocess import extract_patch

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "SPIE-AAPM Lung CT Challenge"
HIT_MM = 10.0  # a detection counts as finding the nodule if its centre is within this distance


def read_nodules(calibration, test):
    rows = []
    if calibration:
        c = pd.read_excel(calibration).dropna(subset=["Nodule Center Image"])
        for _, r in c.iterrows():
            pos = str(r["Nodule Center x,y Position*"])
            m = re.match(r"^\s*(\d+)\s*,\s*(\d+)", pos)
            if m:
                x, y = int(m.group(1)), int(m.group(2))
            else:  # stored as one number: "120, 325" -> 120325 (y always has 3 digits)
                v = int(float(pos))
                x, y = v // 1000, v % 1000
            rows.append(dict(set="calibration", scan=str(r["Scan Number"]).upper().replace("CT-TRAINING-", "CT-Training-"),
                             nodule=1, x=x, y=y, image=int(r["Nodule Center Image"]),
                             diagnosis=str(r["Diagnosis"]).strip().lower()))
    if test:
        t = pd.read_excel(test).dropna(subset=["Nodule Center Image"])
        for _, r in t.iterrows():
            x, y = (int(v) for v in re.findall(r"\d+", str(r["Nodule Center x,y Position*"]))[:2])
            rows.append(dict(set="test", scan=str(r["Scan Number"]).strip(), nodule=int(r["Nodule Number"]), x=x, y=y,
                             image=int(r["Nodule Center Image"]), diagnosis=str(r["Final Diagnosis"]).strip().lower()))
    df = pd.DataFrame(rows)
    df["label"] = df.diagnosis.map(lambda d: 1 if d in ("malignant", "primary lung cancer")
                                   else 0 if d in ("benign", "benign nodule") else -1)
    return df


def load_zip_series(data):
    """Largest CT series in a zip -> (volume HU (row, col, slice) inferior->superior, spacing, instance numbers)."""
    series = {}
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            try:
                ds = pydicom.dcmread(io.BytesIO(zf.read(name)), force=True)
            except Exception:
                continue
            if "PixelData" in ds and "ImagePositionPatient" in ds:
                series.setdefault(str(ds.SeriesInstanceUID), []).append(ds)
    slices = max(series.values(), key=len)
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    vol = np.stack([s.pixel_array * float(getattr(s, "RescaleSlope", 1)) + float(getattr(s, "RescaleIntercept", 0))
                    for s in slices], axis=-1).astype(np.float32)
    z = [float(s.ImagePositionPatient[2]) for s in slices]
    spacing = np.array([float(slices[0].PixelSpacing[0]), float(slices[0].PixelSpacing[1]), float(np.median(np.diff(z)))])
    return vol, spacing, [int(getattr(s, "InstanceNumber", i + 1)) for i, s in enumerate(slices)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/lungx"))
    parser.add_argument("--detect", action="store_true", help="also run the automatic detector on every scan")
    parser.add_argument("--bundle", type=Path, default=Path("data/bundles/lung_nodule_ct_detection"))
    parser.add_argument("--limit", type=int, default=0, help="only the first N scans (for a quick try)")
    args = parser.parse_args()
    names = args.names or [r.name for r in args.runs]
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nodules = read_nodules(args.calibration, args.test)
    models = {n: load_models(r, "final", device) for n, r in zip(names, args.runs)}
    detector = None
    if args.detect:
        from cadc.detect import detect, load_detector
        detector = load_detector(args.bundle, device)

    series = requests.get(f"{API}/getSeries", params={"Collection": COLLECTION, "Modality": "CT"}, timeout=60).json()
    by_patient = {}
    for s in series:
        key = s["PatientID"].upper()
        if key not in by_patient or s.get("ImageCount", 0) > by_patient[key].get("ImageCount", 0):
            by_patient[key] = s

    done_file = args.out / "nodule_scores.csv"
    done = pd.read_csv(done_file) if done_file.exists() else pd.DataFrame()
    finished = set(done.scan) if len(done) else set()
    scans = [s for s in nodules.scan.unique() if s not in finished]
    if args.limit:
        scans = scans[: args.limit]
    print(f"{nodules.scan.nunique()} scans / {len(nodules)} nodules; {len(finished)} scans already done; {len(scans)} to go", flush=True)

    for i, scan in enumerate(scans, 1):
        t0 = time.time()
        meta = by_patient.get(scan.upper())
        if meta is None:
            print(f"[{i}/{len(scans)}] {scan}: not found on TCIA", flush=True)
            continue
        try:
            r = requests.get(f"{API}/getImage", params={"SeriesInstanceUID": meta["SeriesInstanceUID"]}, timeout=900)
            r.raise_for_status()
            vol, spacing, instances = load_zip_series(r.content)
        except Exception as e:
            print(f"[{i}/{len(scans)}] {scan}: download/read failed: {e}", flush=True)
            continue
        cands = detect(detector, vol, spacing, device, min_score=0.3) if detector is not None else None

        rows = []
        for _, n in nodules[nodules.scan == scan].iterrows():
            # The spreadsheet gives the image (instance) number and 1-based pixel coordinates.
            k = instances.index(n.image) if n.image in instances else int(np.clip(n.image - 1, 0, vol.shape[2] - 1))
            row, col = n.y - 1, n.x - 1
            patch = extract_patch(vol, spacing, np.array([row, col, k]))
            rec = dict(n.to_dict(), slice=k, center_hu=float(vol[max(0, row - 1):row + 2, max(0, col - 1):col + 2, k].mean()))
            probs = []
            for name, ms in models.items():
                p = float(score(ms, patch[None], device)[0][0])
                rec[f"prob_{name}"] = p
                probs.append(p)
            rec["prob_Ensemble"] = float(np.mean(probs))
            if cands is not None:
                dists = [np.linalg.norm([(c["y"] - row) * spacing[0], (c["x"] - col) * spacing[1], (c["slice"] - k) * spacing[2]])
                         for c in cands]
                j = int(np.argmin(dists)) if dists else -1
                rec["detected"] = bool(dists) and dists[j] <= HIT_MM
                rec["nearest_detection_mm"] = float(dists[j]) if dists else None
                rec["detections_in_scan"] = len(cands)
            rows.append(rec)
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(done_file, index=False)
        msg = ", ".join(f"{r['diagnosis']} -> {r['prob_Ensemble']:.0%}" + (" (found)" if r.get("detected") else " (not found)" if "detected" in r else "")
                        for r in rows)
        print(f"[{i}/{len(scans)}] {scan}: {msg}  [{time.time() - t0:.0f}s]", flush=True)

    # ---- summary ----
    ev = done[done.label >= 0]
    result = {"nodules_scored": int(len(done)), "nodules_with_confirmed_label": int(len(ev)),
              "malignant": int(ev.label.sum()), "benign": int((ev.label == 0).sum()), "models": {}}
    for name in list(models) + ["Ensemble"]:
        col = f"prob_{name}"
        if col in ev and ev.label.nunique() == 2:
            result["models"][name] = summarize(ev.label, ev[col])
    if "detected" in done:
        det = done[done.label >= 0]
        result["detection"] = dict(found=int(det.detected.sum()), total=int(len(det)),
                                   recall=float(det.detected.mean()),
                                   mean_detections_per_scan=float(done.groupby("scan").detections_in_scan.first().mean()))
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"\nLUNGx external test: {len(ev)} nodules with confirmed diagnosis ({result['malignant']} malignant, {result['benign']} benign)")
    for name, m in result["models"].items():
        print(f"  {name:<18} AUC {m['auc']:.3f} (95% CI {m['auc_95ci'][0]:.3f}-{m['auc_95ci'][1]:.3f}) | acc {m['accuracy']:.3f} | "
              f"sens {m['sensitivity']:.3f} | spec {m['specificity']:.3f}")
    if "detection" in result:
        d = result["detection"]
        print(f"  detector found {d['found']}/{d['total']} nodules ({d['recall']:.0%}), "
              f"{d['mean_detections_per_scan']:.1f} detections per scan on average")
    print(f"saved {done_file} and {args.out / 'metrics.json'}")


if __name__ == "__main__":
    main()
