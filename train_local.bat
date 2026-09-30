@echo off
rem Train all 5 folds on this PC's GPU, then evaluate.
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
    .venv\Scripts\python -u -m cadc.train --data data\patches --out runs\baseline --fold %%f --batch-size 16 --workers 4
    if errorlevel 1 (
        echo Training fold %%f failed.
        pause
        exit /b 1
    )
)

.venv\Scripts\python -m cadc.evaluate --run runs\baseline
echo.
echo Done. Results: runs\baseline\metrics.json and runs\baseline\roc.png
pause
