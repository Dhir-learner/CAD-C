@echo off
rem Train both models (3D ResNet and 2.5D multi-view) on all 5 folds on this PC's GPU,
rem then evaluate each, build the ensemble, and test against confirmed diagnoses.
rem Usage: double-click, or run "train_local.bat" in the CADC folder.
rem Resumable: if interrupted, run it again and it continues where it stopped.
cd /d "%~dp0"

if not exist data\patches\*.npz (
    echo No patches found. Unzip patches.zip so the .npz files are in data\patches\
    pause
    exit /b 1
)

.venv\Scripts\python -c "import torch, sys; ok = torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0) if ok else 'NOT FOUND'); sys.exit(0 if ok else 1)"
if errorlevel 1 (
    echo PyTorch cannot see the GPU. See "Running on your own PC" in README.md.
    pause
    exit /b 1
)

for /L %%f in (0,1,4) do (
    .venv\Scripts\python -u -m cadc.train --arch resnet3d --data data\patches --out runs\baseline --fold %%f --batch-size 16 --workers 4
    if errorlevel 1 goto failed
)
for /L %%f in (0,1,4) do (
    .venv\Scripts\python -u -m cadc.train --arch multiview2d --data data\patches --out runs\multiview --fold %%f --batch-size 32 --workers 4
    if errorlevel 1 goto failed
)

.venv\Scripts\python -m cadc.evaluate --run runs\baseline
.venv\Scripts\python -m cadc.evaluate --run runs\multiview
if exist data\meta\diagnosis.xls (
    .venv\Scripts\python -m cadc.diagnosis_eval --runs runs\baseline --data data\patches --diagnosis data\meta\diagnosis.xls
    .venv\Scripts\python -m cadc.diagnosis_eval --runs runs\multiview --data data\patches --diagnosis data\meta\diagnosis.xls
    .venv\Scripts\python -m cadc.diagnosis_eval --runs runs\baseline runs\multiview --out runs\ensemble --data data\patches --diagnosis data\meta\diagnosis.xls
)
.venv\Scripts\python -m cadc.ensemble --runs runs\baseline runs\multiview --names "3D ResNet" "2.5D multi-view" --out runs\ensemble
echo.
echo Done. Results: runs\ensemble\comparison.json, metrics.json and roc.png
pause
exit /b 0

:failed
echo Training failed; see the messages above.
pause
exit /b 1
