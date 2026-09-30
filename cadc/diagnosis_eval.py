"""Check the models against real diagnoses instead of radiologist opinions.

For 157 LIDC-IDRI patients the sites recorded a confirmed diagnosis (benign, primary lung
cancer, or metastasis), established by biopsy, surgery, 2 years of stability, or progression
(tcia-diagnosis-data-2012-04-20.xls from The Cancer Imaging Archive).

Each patient is scored by their most suspicious nodule. Every nodule is scored by the fold model
that never trained on that patient (patients whose nodules were all too ambiguous to train on are
scored by all fold models). As a baseline, the same patients are scored by the radiologists'
highest mean malignancy rating.

    python -m cadc.diagnosis_eval --runs runs/baseline --data data/patches --diagnosis data/meta/diagnosis.xls
    python -m cadc.diagnosis_eval --runs runs/baseline runs/multiview ...   (ensemble: averages the runs)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cadc.data import assign_folds, load_nodules
from cadc.evaluate import summarize
from cadc.predict import load_models, score

LABELS = {1: "benign", 2: "primary lung cancer", 3: "metastatic"}
METHODS = {0: "unknown", 1: "2-year stability", 2: "biopsy", 3: "surgical resection", 4: "progression or response"}


def read_diagnoses(path):
    d = pd.read_excel(path)
    d = d.iloc[:, :4]
    d.columns = ["patient_id", "diagnosis", "method", "primary_site"]
    d = d[d.diagnosis.isin([1, 2, 3])].copy()
    d["malignant"] = (d.diagnosis >= 2).astype(int)
    return d


def all_nodules(shard_dir, patients):
    """Every nodule of the given patients, including ambiguous ones: (patches, table)."""
    rows, patches = [], []
    for path in sorted(Path(shard_dir).glob("*.npz")):
        pid = path.name.rsplit("_", 1)[0]
        if pid not in patients:
            continue
        with np.load(path) as z:
            for i, r in enumerate(z["malignancy"]):
                r = r[r > 0]
                rows.append(dict(patient_id=pid, shard=path.stem, nodule=i, mean_rating=float(r.mean())))
            if len(z["malignancy"]):
                patches.append(z["patches"])
    return (np.concatenate(patches) if patches else np.zeros((0, 64, 64, 64), np.int16)), pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, type=Path, help="one run, or several to average")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--diagnosis", required=True, type=Path)
    parser.add_argument("--which", choices=["final", "best"], default="final")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, help="where to write diagnosis_metrics.json and the per-patient CSV "
                                                 "(default: the first run directory)")
    args = parser.parse_args()
    out = args.out or args.runs[0]
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dx = read_diagnoses(args.diagnosis)
    patches, nod = all_nodules(args.data, set(dx.patient_id))

    # Rebuild the training split: same data, seed and fold count give the same folds.
    _, train_table = load_nodules(args.data, 1)
    fold_of = assign_folds(train_table, args.folds, args.seed).groupby("patient_id").fold.first().to_dict()

    probs = []
    for run in args.runs:
        models = load_models(run, args.which, device)
        p = np.zeros(len(nod))
        for pid, idx in nod.groupby("patient_id").groups.items():
            idx = np.asarray(list(idx))
            use = [models[fold_of[pid]]] if pid in fold_of else models
            p[idx] = np.mean([score([m], patches[idx], device)[0] for m in use], axis=0)
        probs.append(p)
    nod["prob"] = np.mean(probs, axis=0)

    per_patient = nod.groupby("patient_id").agg(model=("prob", "max"), radiologists=("mean_rating", "max"),
                                                nodules=("prob", "size"))
    df = dx.merge(per_patient, left_on="patient_id", right_index=True, how="left")
    no_nodules = int(df.model.isna().sum())
    df = df.dropna(subset=["model"])
    df.to_csv(out / "diagnosis_predictions.csv", index=False)

    rad = summarize(df.malignant, (df.radiologists - 1) / 4)  # ratings 1-5 mapped to 0-1; 3.5 -> 0.625
    result = {
        "runs": [str(r) for r in args.runs],
        "patients_with_diagnosis": int(len(dx)),
        "patients_scored": int(len(df)),
        "patients_without_nodules": no_nodules,
        "diagnoses": {LABELS[k]: int(v) for k, v in df.diagnosis.value_counts().sort_index().items()},
        "methods": {METHODS.get(int(k), str(k)): int(v) for k, v in df.method.value_counts().sort_index().items()},
        "model": summarize(df.malignant, df.model),
        "radiologists": {k: rad[k] for k in ("n", "n_positive", "auc", "auc_95ci")},
    }
    prim = df[df.diagnosis != 3]
    if prim.malignant.nunique() == 2:
        result["model_primary_vs_benign"] = summarize(prim.malignant, prim.model)
        result["radiologists_primary_vs_benign_auc"] = summarize(prim.malignant, prim.radiologists)["auc"]
    (out / "diagnosis_metrics.json").write_text(json.dumps(result, indent=2))

    m = result["model"]
    print(f"patients: {len(df)} scored ({result['diagnoses']}), {no_nodules} without nodules >= 3 mm")
    print(f"diagnosis methods: {result['methods']}")
    print(f"model:        AUC {m['auc']:.3f} (95% CI {m['auc_95ci'][0]:.3f}-{m['auc_95ci'][1]:.3f}) | "
          f"acc {m['accuracy']:.3f} | sens {m['sensitivity']:.3f} | spec {m['specificity']:.3f}")
    r = result["radiologists"]
    print(f"radiologists: AUC {r['auc']:.3f} (95% CI {r['auc_95ci'][0]:.3f}-{r['auc_95ci'][1]:.3f})  [highest mean rating]")
    if "model_primary_vs_benign" in result:
        print(f"primary lung cancer vs benign only: model AUC {result['model_primary_vs_benign']['auc']:.3f}, "
              f"radiologists AUC {result['radiologists_primary_vs_benign_auc']:.3f}")
    print(f"saved {out / 'diagnosis_metrics.json'} and diagnosis_predictions.csv")


if __name__ == "__main__":
    main()
