"""Combine the out-of-fold predictions of several runs (e.g. 3D ResNet + 2.5D multi-view) and compare.

All runs use the same patient folds, so averaging their out-of-fold probabilities gives an honest
ensemble estimate: every nodule is still scored only by models that never trained on its patient.

    python -m cadc.ensemble --runs runs/baseline runs/multiview --out runs/ensemble

Writes into --out: oof_predictions.csv, metrics.json (same format as cadc.evaluate), roc.png and
comparison.json (each run next to the ensemble).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from cadc.evaluate import summarize

KEY = ["patient_id", "shard", "nodule"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--names", nargs="+", help="display names, one per run")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    names = args.names or [r.name for r in args.runs]
    args.out.mkdir(parents=True, exist_ok=True)

    tables = []
    for run in args.runs:
        f = run / "oof_predictions.csv"
        if not f.exists():
            raise SystemExit(f"{f} not found: run cadc.evaluate on {run} first")
        tables.append(pd.read_csv(f))
    merged = tables[0].rename(columns={"prob": "prob_0"})
    for i, t in enumerate(tables[1:], start=1):
        merged = merged.merge(t[KEY + ["prob"]].rename(columns={"prob": f"prob_{i}"}), on=KEY, how="inner")
    if len(merged) != len(tables[0]):
        raise SystemExit("runs cover different nodules; were they trained on the same data and folds?")
    prob_cols = [f"prob_{i}" for i in range(len(tables))]
    merged["prob"] = merged[prob_cols].mean(axis=1)
    merged.to_csv(args.out / "oof_predictions.csv", index=False)

    def levels(df, col):
        pat = df.groupby("patient_id").agg(label=("label", "max"), prob=(col, "max"))
        return summarize(df.label, df[col]), summarize(pat.label, pat.prob)

    comparison = []
    for name, col in list(zip(names, prob_cols)) + [("Ensemble", "prob")]:
        nod, pat = levels(merged, col)
        row = dict(name=name, nodule_level=nod, patient_level=pat)
        dx = (args.runs[names.index(name)] if name in names else args.out) / "diagnosis_metrics.json"
        if dx.exists():
            row["diagnosis_auc"] = json.loads(dx.read_text())["model"]["auc"]
        comparison.append(row)

    ens_nod, ens_pat = levels(merged, "prob")
    metrics = {
        "predictions": "final",
        "runs": [str(r) for r in args.runs],
        "folds": sorted(merged.fold_dir.unique().tolist()),
        "per_fold_auc": {d: float(roc_auc_score(g.label, g.prob)) for d, g in merged.groupby("fold_dir")},
        "nodule_level": ens_nod,
        "patient_level": ens_pat,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.out / "comparison.json").write_text(json.dumps(comparison, indent=2))

    fig, ax = plt.subplots(figsize=(5, 5))
    for row, col in zip(comparison, prob_cols + ["prob"]):
        fpr, tpr, _ = roc_curve(merged.label, merged[col])
        ax.plot(fpr, tpr, lw=2.5 if col == "prob" else 1.2, label=f"{row['name']} AUC = {row['nodule_level']['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Out-of-fold ROC, nodule level")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out / "roc.png", dpi=120)

    print(f"{'model':<22}{'nodule AUC':>12}{'95% CI':>16}{'accuracy':>10}{'patient AUC':>13}{'diagnosis AUC':>15}")
    for r in comparison:
        n, p = r["nodule_level"], r["patient_level"]
        ci = f"{n['auc_95ci'][0]:.3f}-{n['auc_95ci'][1]:.3f}"
        dx = f"{r['diagnosis_auc']:.3f}" if "diagnosis_auc" in r else "-"
        print(f"{r['name']:<22}{n['auc']:>12.3f}{ci:>16}{n['accuracy']:>10.3f}{p['auc']:>13.3f}{dx:>15}")
    print(f"saved metrics.json, comparison.json, oof_predictions.csv, roc.png in {args.out}")


if __name__ == "__main__":
    main()
