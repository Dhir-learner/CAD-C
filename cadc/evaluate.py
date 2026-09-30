"""Pool the out-of-fold predictions of a cross-validation run and report results.

Every nodule is predicted by the one fold model that never trained on its
patient, so pooling the folds gives one honest prediction per nodule.

    python -m cadc.evaluate --run /content/drive/MyDrive/CADC/runs/baseline

Writes metrics.json, roc.png and oof_predictions.csv into the run directory.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def summarize(labels, probs, n_boot=1000, seed=0):
    """AUC with a bootstrap 95% CI, plus accuracy, sensitivity, specificity at 0.5."""
    labels, probs = np.asarray(labels), np.asarray(probs)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        if labels[idx].min() != labels[idx].max():
            boots.append(roc_auc_score(labels[idx], probs[idx]))
    pred = probs >= 0.5
    return dict(
        n=int(len(labels)),
        n_positive=int(labels.sum()),
        auc=float(roc_auc_score(labels, probs)),
        auc_95ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        accuracy=float((pred == labels).mean()),
        sensitivity=float(pred[labels == 1].mean()),
        specificity=float((~pred[labels == 0]).mean()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path, help="run directory containing fold*/ subfolders")
    parser.add_argument("--which", choices=["final", "best"], default="final",
                        help="final = last-epoch predictions (unbiased, default); "
                             "best = best-epoch predictions (optimistic, chosen on the same data)")
    args = parser.parse_args()

    name = "val_predictions_final.csv" if args.which == "final" else "val_predictions.csv"
    files = sorted(args.run.glob(f"fold*/{name}"))
    if not files:
        raise SystemExit(f"no fold*/{name} in {args.run}; has training finished?")
    oof = pd.concat([pd.read_csv(f).assign(fold_dir=f.parent.name) for f in files], ignore_index=True)
    oof.to_csv(args.run / "oof_predictions.csv", index=False)

    # A patient counts as positive if any nodule is malignant; their score is their most suspicious nodule.
    patients = oof.groupby("patient_id").agg(label=("label", "max"), prob=("prob", "max"))

    result = {
        "predictions": args.which,
        "folds": [f.parent.name for f in files],
        "per_fold_auc": {d: float(roc_auc_score(g.label, g.prob)) for d, g in oof.groupby("fold_dir")},
        "nodule_level": summarize(oof.label, oof.prob),
        "patient_level": summarize(patients.label, patients.prob),
    }
    (args.run / "metrics.json").write_text(json.dumps(result, indent=2))

    fig, ax = plt.subplots(figsize=(5, 5))
    for label, df in [("nodule", oof), ("patient", patients)]:
        fpr, tpr, _ = roc_curve(df.label, df.prob)
        auc = result[f"{label}_level"]["auc"]
        ax.plot(fpr, tpr, label=f"{label}-level AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title=f"Out-of-fold ROC ({args.which} epoch)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.run / "roc.png", dpi=120)

    print(f"folds: {', '.join(result['folds'])}")
    for d, auc in result["per_fold_auc"].items():
        print(f"  {d}: AUC {auc:.4f}")
    for level in ("nodule_level", "patient_level"):
        m = result[level]
        print(f"{level.replace('_', ' ')}: n={m['n']} ({m['n_positive']} malignant) | "
              f"AUC {m['auc']:.4f} (95% CI {m['auc_95ci'][0]:.3f}-{m['auc_95ci'][1]:.3f}) | "
              f"acc {m['accuracy']:.3f} | sens {m['sensitivity']:.3f} | spec {m['specificity']:.3f}")
    print(f"saved metrics.json, roc.png, oof_predictions.csv in {args.run}")


if __name__ == "__main__":
    main()
