# CADC: lung nodule malignancy prediction on LIDC-IDRI

An ensemble of two deep-learning models (a 3D CNN and a 2.5D multi-view CNN) that looks at a
lung nodule in a CT scan and estimates how likely it is to be malignant. A pretrained detector
can also find nodules automatically. It is trained on the public LIDC-IDRI dataset (1,010 patients,
1,018 CT scans), with a local web app for exploring it.

![Tour of the web app](docs/demo.gif)

**What the labels are.** In LIDC-IDRI, up to 4 radiologists rated each nodule's
likelihood of malignancy from 1 (highly unlikely) to 5 (highly suspicious).
This project averages the ratings per nodule: at most 2.5 counts as benign,
at least 3.5 counts as malignant, and nodules in between are left out as
ambiguous. These are expert opinions, not biopsy results, so the model learns
to predict what radiologists would think. It is a research project, not a
clinical tool.

## Results

All numbers come from 5-fold cross-validation split by patient: each nodule is scored only by
models that never saw that patient. 1,627 nodules from 724 patients (504 malignant).

| Model | AUC vs radiologist ratings (95% CI) | Accuracy | Sensitivity | Specificity | Patient-level AUC | AUC vs confirmed diagnosis |
|---|---|---|---|---|---|---|
| 3D ResNet (8.3 M params, from scratch) | 0.924 (0.907–0.939) | 86.7% | 83.9% | 87.9% | 0.933 | 0.662 |
| 2.5D multi-view (ResNet18, ImageNet-pretrained) | 0.931 (0.916–0.944) | 86.4% | 83.3% | 87.8% | 0.929 | 0.713 |
| **Ensemble (10 networks)** | **0.936 (0.921–0.949)** | **88.3%** | **84.9%** | **89.8%** | **0.941** | 0.680 |
| *Radiologists' own ratings* | – | – | – | – | – | *0.765* |

- **Against radiologists' ratings** the ensemble beats either model alone, within the 0.85–0.95 range
  usually reported for LIDC-IDRI.
- **Against confirmed diagnoses** (118 LIDC patients with a diagnosis confirmed by biopsy, surgery,
  2-year stability or progression), every model scores much lower (AUC about 0.66–0.71), below the
  radiologists' own ratings (0.77). The models learned to imitate radiologists' opinions, and predicting
  actual cancer is harder; many of these patients also had metastases from other cancers. With 118 patients
  the confidence intervals are wide (about ±0.12), so the differences between the three models on this test
  are not meaningful. This is the most honest limitation of the project.
- **No overfitting**: training vs unseen-patient AUC differ by about 0.02, and validation AUC does not fall
  late in training (see the Model page of the web app).
- **Detection**: MONAI's pretrained LUNA16 RetinaNet found 14 of 17 radiologist-marked nodules on a
  spot check of 3 scans (it was trained on LUNA16, which comes from LIDC-IDRI, so this is not an independent test).

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

## Second model, ensemble and diagnosis check

Besides the 3D ResNet (`--arch resnet3d`, the default), `cadc.train` can train a **2.5D multi-view
CNN** (`--arch multiview2d`): it slices 9 planes through the nodule (3 orthogonal, 6 diagonal), runs each
through one shared ImageNet-pretrained ResNet18, and pools the features. Both use the same patient folds,
so their out-of-fold predictions can be averaged honestly.

```
rem everything in one go (both models, evaluation, ensemble, diagnosis check):
train_local.bat

rem or step by step:
.venv\Scripts\python -m cadc.train --arch multiview2d --data data\patches --out runs\multiview --fold 0
.venv\Scripts\python -m cadc.evaluate --run runs\multiview
.venv\Scripts\python -m cadc.ensemble --runs runs\baseline runs\multiview --names "3D ResNet" "2.5D multi-view" --out runs\ensemble
.venv\Scripts\python -m cadc.diagnosis_eval --runs runs\baseline runs\multiview --out runs\ensemble --data data\patches --diagnosis data\meta\diagnosis.xls
```

The diagnosis spreadsheet comes from The Cancer Imaging Archive:
`curl -L -o data\meta\diagnosis.xls https://www.cancerimagingarchive.net/wp-content/uploads/tcia-diagnosis-data-2012-04-20.xls`

## Automatic nodule detection

The web app can search a whole scan for nodules with MONAI's pretrained
[lung_nodule_ct_detection](https://github.com/Project-MONAI/model-zoo) bundle (RetinaNet trained on LUNA16,
Apache-2.0), then score every candidate with the ensemble. One-time setup (about 160 MB):

```
uv pip install -r requirements-app.txt
.venv\Scripts\python -c "from monai.bundle import download; download(name='lung_nodule_ct_detection', bundle_dir='data/bundles')"
```

It takes about 30 seconds per scan on the laptop GPU. Without the bundle the app still works; the button is hidden.

## AI chat assistant

Every page of the web app has an **Ask CADC AI** button: a chat assistant (via the [Groq API](https://console.groq.com),
model `openai/gpt-oss-120b`) that explains lung nodules, lung cancer, the model's results and how to use the app. On the
Analyse page it knows your latest result, so you can ask *"What does my result mean?"*.

It is instructed never to diagnose, to always point people to a doctor, and to put emergency advice first when someone
describes emergency symptoms. Chat messages are sent to Groq to generate replies (scans are not).

**Setup:** create a file named `.env` in the CADC folder containing your key:

```
GROQ_API_KEY=your-key-here
```

`.env` is in `.gitignore`, so the key never goes to GitHub. **Never put the key in code**; this repository is public.
Optionally add `GROQ_MODEL=...` to use a different Groq model. Without a key the app works, and the chat button shows it isn't configured.

## Web app

A local website with five pages. It runs on your PC with the trained models in `runsaseline`.

| Page | What's there |
|---|---|
| **Home** | Overview, live headline results, an interactive 3D model of the lungs |
| **Learn** | What lung cancer is, types, risk factors, symptoms, lung nodules, staging, diagnosis, treatment, prevention, and *when to see a doctor*. Includes a 3D explorer: click labelled parts of the lungs, and switch between healthy and stages I–IV to watch a tumour grow, reach the lymph nodes and spread |
| **Analyse a scan** | Upload a chest CT, let the AI **find nodules automatically** or click one, get the 10-network ensemble estimate with a low / intermediate / high band, each model's vote, CT views and an **AI attention heatmap** (Grad-CAM), plus a printable report |
| **The model** | Accuracy, sensitivity, specificity, interactive ROC and learning curves, model comparison, the real-diagnosis test, per-fold results, architecture and limitations |
| **About & FAQ** | Medical disclaimer, privacy, FAQ, credits |

**AI warnings.** On first visit every page shows a disclaimer that must be accepted (it's an AI research tool, not a diagnosis; consult a doctor).
The analyser stays locked until it is accepted, and every result repeats the advice to see a doctor.

**Start it:** double-click **`run_app.bat`** in the CADC folder. The browser opens at http://127.0.0.1:8000
(keep the black window open while using it). Scans are processed on your PC and never uploaded anywhere.

**Analyse a scan:**
1. Drop a **.zip of a DICOM CT series** (or select all its `.dcm` files), or click **Try a sample scan**.
2. Scroll through the slices (mouse wheel, slider, or arrow keys), find the nodule where it looks largest, and **click its centre**.
3. Click **Analyse nodule** (or press Enter). Switch between **CT** and **AI attention** to see where the models looked.

For LIDC-IDRI scans, the nodules the radiologists marked are listed under the viewer: click one to analyse it directly.
Those scans were in the training data, so their scores are optimistic (the app says so).
The app does **not** find nodules by itself: you point to them.

First-time setup on a new PC (already done on this one):
```
uv pip install -r requirements.txt -r requirements-app.txt
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
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
| `cadc/model.py` | The 3D ResNet (8.3 M parameters) and the 2.5D multi-view CNN (ImageNet-pretrained ResNet18, 11.2 M). |
| `cadc/ensemble.py` | Averages several runs' out-of-fold predictions and compares them with the ensemble. |
| `cadc/diagnosis_eval.py` | Tests models against the 157 LIDC patients with confirmed diagnoses, next to the radiologists' ratings. |
| `cadc/detect.py` | Finds nodule candidates in a whole scan with MONAI's pretrained LUNA16 detector. |
| `docs/make_demo_gif.py` | Records `docs/demo.gif` from the running web app. |
| `cadc/train.py` | Trains one fold with mixed precision, class-balanced loss and a cosine learning-rate schedule. Saves `last.pt` every epoch (for resuming), `best.pt` and `final.pt`, `history.csv` and validation predictions. |
| `cadc/evaluate.py` | Pools the 5 folds' predictions and reports AUC with a 95% confidence interval, accuracy, sensitivity, specificity, nodule and patient level, plus a ROC curve. |
| `cadc/predict.py` | Averages the 5 models' scores on nodules from a shard or a new DICOM scan. |
| `notebooks/colab_launcher.ipynb` | The Colab notebook used above. |
| `app/server.py` | Web app backend (FastAPI): pages, scan upload, predictions, Grad-CAM, detection, metrics and chat APIs. |
| `app/chat.py` | The AI assistant: system prompt with safety rules and real results, streaming calls to Groq. |
| `app/static/` | Web app frontend: 5 pages, shared CSS/JS, the Three.js 3D lung model (`js/lungs3d.js`), charts (`js/model.js`). |
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
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m cadc.train --data data\patches --out runs\local --fold 0 --batch-size 16
```

(Copy `MyDrive/CADC/patches` to `data\patches` first.)

## Data and licence

LIDC-IDRI is provided by The Cancer Imaging Archive under CC BY 3.0. Please
cite: Armato SG III et al., *The Lung Image Database Consortium (LIDC) and Image
Database Resource Initiative (IDRI)*, Medical Physics 38(2), 2011.
