# CADC: lung nodule malignancy prediction on LIDC-IDRI

A 3D convolutional neural network that looks at a lung nodule in a CT scan and
estimates how likely it is to be malignant. It is trained on the public
LIDC-IDRI dataset (1,010 patients, 1,018 CT scans) on Google Colab.

**What the labels are.** In LIDC-IDRI, up to 4 radiologists rated each nodule's
likelihood of malignancy from 1 (highly unlikely) to 5 (highly suspicious).
This project averages the ratings per nodule: at most 2.5 counts as benign,
at least 3.5 counts as malignant, and nodules in between are left out as
ambiguous. These are expert opinions, not biopsy results, so the model learns
to predict what radiologists would think. It is a research project, not a
clinical tool.

---

## How to run it (step by step)

Everything runs in Google Colab from one notebook. You don't need to install
anything on your PC.

### 0. Open the notebook

Open this link while signed in to your Google account:

**https://colab.research.google.com/github/Dhir-learner/CAD-C/blob/main/notebooks/colab_launcher.ipynb**

(Or in Colab: *File → Open notebook → GitHub*, paste
`https://github.com/Dhir-learner/CAD-C`, and pick `notebooks/colab_launcher.ipynb`.)

Optionally, use *File → Save a copy in Drive* so your outputs stay in the notebook.

### Setup cell (run this first, every time)

Run the **Setup** cell. It:
1. connects Google Drive (approve the popup; your results are saved there),
2. downloads the latest code from GitHub,
3. installs the Python packages.

Run it again whenever you open the notebook, reconnect, or change the runtime.

### Step 1: Preprocess the data (once; about 2–3 hours; no GPU needed)

> **If you already started preprocessing earlier** (e.g. in another notebook),
> let that finish first. Then just run the check cell in Step 1: if it says
> `1018/1018`, skip to Step 2.

1. Use a **CPU runtime** (*Runtime → Change runtime type → CPU*). It doesn't use GPU quota.
2. Run **Setup**, then the preprocessing cell. For each of the 1,018 scans, it
   downloads the CT from the Cancer Imaging Archive, cuts out a 64 mm cube around
   every nodule, saves it to `MyDrive/CADC/patches/`, and deletes the raw scan.
   It prints one line per scan.
3. **Keep the browser tab open and stop the PC from sleeping** (Windows:
   *Settings → System → Power → Sleep: Never*) while it runs.
4. When it finishes, run the check cell. It should say `1018/1018`.

**If it disconnects:** reconnect, run Setup, and run the preprocessing cell
again. Finished scans are skipped. The same goes for any scan listed in
`failed.log`: rerunning retries only those.

### Step 2: Train (about 1–2 hours in total; needs a GPU)

1. Switch to a GPU: *Runtime → Change runtime type → T4 GPU → Save*.
   (This restarts the machine; your files on Drive are safe.)
2. Run **Setup** again. The last line should show `Tesla T4`.
3. Run the **copy** cell (copies the patches from Drive to the fast local disk).
4. Run the **training** cell. It trains 5 models, one per cross-validation fold
   (see "How the evaluation works" below), 60 epochs each. Every epoch prints a line like:
   ```
   epoch=12 | train_loss=0.41 | lr=0.0009 | auc=0.91 | accuracy=0.86 | ... | best_auc=0.9204
   ```
   `auc` is the key number: 0.5 is guessing, 1.0 is perfect.

**If it disconnects or the free GPU quota runs out** (Colab's free tier allows
a few GPU hours per day): come back later, switch to T4, run Setup → copy →
training again. Each fold resumes from its last saved epoch, and finished folds
are skipped automatically.

### Step 3: Evaluate

Run the **evaluate** cell. It prints the results and shows a ROC curve:

```
fold0: AUC ...            <- each of the 5 models on its own held-out patients
nodule level: n=... | AUC ... (95% CI ...) | acc ... | sens ... | spec ...
patient level: n=... | AUC ...
```

- **Nodule level:** how well it separates malignant from benign nodules.
- **Patient level:** a patient counts as positive if any of their nodules is malignant,
  and is scored by their most suspicious nodule.
- **sens / spec:** the share of malignant / benign nodules classified correctly at the 0.5 threshold.

Results are saved to `MyDrive/CADC/runs/baseline/`: `metrics.json`, `roc.png`,
and `oof_predictions.csv` (one prediction per nodule).

For reference, published models on LIDC-IDRI with this kind of labelling
usually report a nodule-level AUC of roughly 0.85–0.95.

### Step 4: Predict

The **predict** cell scores the nodules of one preprocessed scan with all 5 models:

```
python -m cadc.predict --run RUN --shard PATCHES/LIDC-IDRI-0001_0012.npz
```

To score a nodule in **a new CT scan** (a folder of `.dcm` files), give its centre as seen in a DICOM viewer:

```
python -m cadc.predict --run RUN --dicom /content/my_scan --center X Y SLICE
```

`X` is the column, `Y` the row, and `SLICE` the slice number counted from the
lowest slice (feet end), all starting at 0. Upload the folder to Colab or Drive first.

---

## Web app: upload a CT scan and analyse a nodule

A local web page where you upload a chest CT, click on a nodule, and get the
5-model malignancy estimate, each model's vote, and axial/coronal/sagittal views
of what the model saw. It runs on your PC with the trained models in `runsaseline`.

**Start it:** double-click **`run_app.bat`** in the CADC folder. The browser opens
at http://127.0.0.1:8000 (keep the black window open while using it).

**Use it:**
1. Drop a **.zip of a DICOM CT series** (or select all its `.dcm` files), or click
   **Try a sample scan** to load LIDC-IDRI-0001.
2. Scroll through the slices (mouse wheel, slider, or arrow keys), find the nodule
   where it looks largest, and **click its centre**.
3. Click **Analyse nodule** (or press Enter).

For LIDC-IDRI scans, the nodules the radiologists marked are listed under the
viewer: click one to analyse it directly. Those scans were in the training data,
so their scores are optimistic (the app says so).

The app does **not** find nodules by itself: you point to them. It is a research
prototype, not a diagnostic tool.

First-time setup on a new PC (already done on this one):
```
uv pip install -r requirements.txt -r requirements-app.txt
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
```
Options: `run_app.bat --run runs\other_run --port 8080`.

## How the evaluation works (why the numbers are honest)

- **5-fold cross-validation by patient.** Patients are split into 5 groups. Each
  model trains on 4 groups and is tested on the 5th, which it has never seen.
  All nodules of one patient stay in the same group, so the model can't
  "recognise" a patient it trained on.
- **Final-epoch predictions.** `evaluate` uses each model's last epoch by
  default. The "best epoch" (highest validation AUC) is also saved, but choosing it
  looks at the test group, so its score is slightly optimistic.
  `python -m cadc.evaluate --run RUN --which best` shows it anyway.

## What the code does

| File | Purpose |
|---|---|
| `cadc/preprocess.py` | Downloads each scan from TCIA; pylidc groups the radiologists' annotations into nodules; saves a 64³ patch (1 mm voxels, clipped to −1000…400 HU) per nodule with its ratings, one `.npz` file per scan. Resumable. |
| `cadc/data.py` | Loads the patches, applies the label rule, makes patient-grouped folds, and augments training data (random 48³ crops, flips, rotations). |
| `cadc/model.py` | A small 3D ResNet (about 8 million parameters) that outputs one malignancy score. |
| `cadc/train.py` | Trains one fold with mixed precision, class-balanced loss and a cosine learning-rate schedule. Saves `last.pt` every epoch (for resuming), `best.pt` and `final.pt`, `history.csv` and validation predictions. |
| `cadc/evaluate.py` | Pools the 5 folds' predictions and reports AUC with a 95% confidence interval, accuracy, sensitivity, specificity, nodule and patient level, plus a ROC curve. |
| `cadc/predict.py` | Averages the 5 models' scores on nodules from a shard or a new DICOM scan. |
| `notebooks/colab_launcher.ipynb` | The Colab notebook used above. |
| `app/server.py`, `app/static/index.html` | The web app (FastAPI backend + single-page frontend). |
| `train_local.bat`, `run_app.bat` | One-click training and web app on Windows. |

Useful options for `cadc.train`: `--epochs`, `--batch-size`, `--lr`, and
`--min-readers 3` (keep only nodules rated by at least 3 radiologists; a common,
stricter choice in papers). To try a variation without overwriting results,
change `RUN` in the Setup cell (e.g. `runs/min3readers`).

## Troubleshooting

| Problem | Fix |
|---|---|
| "Runtime disconnected" | Reconnect, run Setup, rerun the step you were on. Everything resumes. |
| "Cannot connect to GPU backend" / quota reached | The free GPU quota is used up for now. Try again later (often next day); training resumes where it stopped. |
| Training says `device=cpu` | The runtime isn't a GPU. Change the runtime to T4 and rerun Setup. |
| `no .npz shards` | Step 1 hasn't finished, or the copy cell wasn't run in this session. |
| `CUDA out of memory` | Add `--batch-size 16` to the training command. |
| Scans in `failed.log` | Usually a network hiccup. Rerun the preprocessing cell; only those scans are retried. |

## Storage layout (Google Drive)

```
MyDrive/CADC/
├── patches/                one .npz per scan (~1–2 GB total) + failed.log
└── runs/baseline/
    ├── fold0/ … fold4/     last.pt, best.pt, final.pt, history.csv, val_predictions*.csv
    ├── metrics.json, roc.png, oof_predictions.csv   (after Step 3)
```

## Running on your own PC (optional)

Your laptop's RTX 4050 (6 GB) can also train this model.

```
uv venv --python 3.11 .venv
uv pip install -r requirements.txt
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m cadc.train --data data\patches --out runs\local --fold 0 --batch-size 16
```

(Copy `MyDrive/CADC/patches` to `data\patches` first.)

## Data and licence

LIDC-IDRI is provided by The Cancer Imaging Archive under CC BY 3.0. Please
cite: Armato SG III et al., *The Lung Image Database Consortium (LIDC) and Image
Database Resource Initiative (IDRI)*, Medical Physics 38(2), 2011.
