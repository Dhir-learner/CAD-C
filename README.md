# CADC — lung nodule malignancy on LIDC-IDRI

A 3D CNN that classifies lung nodules in CT as likely benign or malignant.
Labels are the LIDC-IDRI radiologists' 1–5 malignancy ratings, averaged per
nodule: at most 2.5 counts as benign, at least 3.5 as malignant, and nodules in
between are dropped. These are expert opinions, not biopsy results.

Training runs on Google Colab GPUs and is driven from Claude Code through the
[Colab MCP server](https://github.com/googlecolab/colab-mcp) configured in `.mcp.json`.

## Layout

| Path | What it does |
|---|---|
| `cadc/preprocess.py` | Downloads each scan from TCIA, cuts a 64 mm cube (1 mm voxels) around every nodule, saves one `.npz` shard per scan, deletes the DICOM. Resumable. |
| `cadc/data.py` | Loads shards, applies the label rule, makes patient-grouped CV folds, augments. |
| `cadc/model.py` | Small 3D ResNet. |
| `cadc/train.py` | Trains one fold with mixed precision; checkpoints every epoch and resumes automatically. |
| `notebooks/colab_launcher.ipynb` | Colab entry point: mount Drive, pull code, preprocess, train, report. |

## Storage

Everything large lives on Google Drive under `MyDrive/CADC/`: `patches/` holds
the shards and `runs/<name>/fold<k>/` the checkpoints, `history.csv` and
`val_predictions.csv`. Nothing large goes in git.

## Local run

```
uv venv --python 3.11 .venv
uv pip install -r requirements.txt torch
.venv/Scripts/python -m cadc.preprocess --out data/patches --work data/dicom --limit 5
.venv/Scripts/python -m cadc.train --data data/patches --out runs/local --fold 0
```
