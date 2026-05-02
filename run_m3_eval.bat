@echo off
REM ======================================================
REM  Milestone 3 Evaluation Script (Windows)
REM
REM  Pipeline:
REM    1. Train logistic regression model (skip if cached)
REM    2. Run lime_m3   - end-to-end GPU LIME (Cholesky)
REM    3. Run lime_m3   - end-to-end GPU LIME (Gradient Descent)
REM    4. CPU NumPy surrogate (Cholesky)        for cross-validation
REM    5. M2 PyTorch surrogate                  for cross-validation
REM    6. Compare all four (GPU/Cholesky, GPU/GD, CPU, M2)
REM    7. Generate plots using explain_instance.py
REM ======================================================
setlocal enabledelayedexpansion

set ROOT=%~dp0
set BUILD=%ROOT%build
set EXE_M1=%BUILD%\Release\lime_m1.exe
set EXE_M3=%BUILD%\Release\lime_m3.exe
set MODELS=%ROOT%models
set PLOTS=%ROOT%plots\m3

set DATASET=breast_cancer
set MODEL_BIN=%MODELS%\%DATASET%.bin
set MODEL_NPZ=%MODELS%\%DATASET%.npz

set B=16384
set RIDGE=1e-3

set XBIN=%BUILD%\X_m3.bin
set PBIN=%BUILD%\preds_m3.bin
set WBIN=%BUILD%\weights_m3.bin

set ATTR_GPU_BIN=%BUILD%\attributions_gpu.bin
set ATTR_GPU_GD_BIN=%BUILD%\attributions_gpu_gd.bin
set ATTR_GPU_NPZ=%BUILD%\attributions_gpu.npz
set ATTR_GPU_GD_NPZ=%BUILD%\attributions_gpu_gd.npz
set ATTR_CPU_NPZ=%BUILD%\attributions_cpu.npz
set ATTR_M2_NPZ=%BUILD%\attributions_m2.npz

if not exist "%BUILD%"   mkdir "%BUILD%"
if not exist "%PLOTS%"   mkdir "%PLOTS%"

echo ========================================
echo  Step 1: Train logistic regression model
echo ========================================
if not exist "%MODEL_BIN%" (
    python "%ROOT%scripts\train_model.py" --dataset=%DATASET% --out-dir="%MODELS%" --instance=30
) else (
    echo Model already present at %MODEL_BIN%, skipping training.
)

echo ========================================
echo  Step 2: GPU LIME end-to-end (Cholesky)
echo ========================================
if not exist "%EXE_M3%" (
    echo [ERROR] %EXE_M3% not found.  Build it first:
    echo     cmake --preset vs2022-cuda
    echo     cmake --build --preset vs2022-cuda-release --target lime_m3
    exit /b 1
)
"%EXE_M3%" --B=%B% --model="%MODEL_BIN%" ^
    --solver=cholesky --ridge=%RIDGE% ^
    --write-X "%XBIN%" ^
    --write-preds "%PBIN%" ^
    --write-weights "%WBIN%" ^
    --write-attributions "%ATTR_GPU_BIN%"

python "%ROOT%scripts\gpu_surrogate_to_npz.py" ^
    --in "%ATTR_GPU_BIN%" --model "%MODEL_NPZ%" --out "%ATTR_GPU_NPZ%"

echo ========================================
echo  Step 3: GPU LIME end-to-end (Gradient Descent)
echo ========================================
"%EXE_M3%" --B=%B% --model="%MODEL_BIN%" ^
    --solver=gd --ridge=%RIDGE% --gd-iters=500 --gd-lr=0.01 ^
    --write-X "%XBIN%" ^
    --write-preds "%PBIN%" ^
    --write-weights "%WBIN%" ^
    --write-attributions "%ATTR_GPU_GD_BIN%"

python "%ROOT%scripts\gpu_surrogate_to_npz.py" ^
    --in "%ATTR_GPU_GD_BIN%" --model "%MODEL_NPZ%" --out "%ATTR_GPU_GD_NPZ%"

echo ========================================
echo  Step 4: CPU NumPy surrogate (Cholesky)
echo ========================================
python "%ROOT%scripts\cpu_surrogate.py" ^
    --X "%XBIN%" --preds "%PBIN%" --weights "%WBIN%" ^
    --model "%MODEL_NPZ%" --B %B% ^
    --solver cholesky --ridge %RIDGE% ^
    --out "%ATTR_CPU_NPZ%"

echo ========================================
echo  Step 5: M2 PyTorch surrogate (framework-assisted)
echo ========================================
set KMP_DUPLICATE_LIB_OK=TRUE
python "%ROOT%scripts\surrogate_train.py" ^
    --X "%XBIN%" --preds "%PBIN%" --weights "%WBIN%" ^
    --model "%MODEL_NPZ%" --B %B% ^
    --epochs 500 --lr 0.01 ^
    --out "%ATTR_M2_NPZ%"

echo ========================================
echo  Step 6: Compare all four surrogates
echo ========================================
python "%ROOT%scripts\compare_surrogates.py" ^
    --m2 "%ATTR_M2_NPZ%" ^
    --m3 "%ATTR_GPU_NPZ%" ^
    --cpu "%ATTR_CPU_NPZ%" ^
    --top-k 10

echo ----- Cholesky vs GD (GPU) -----
python "%ROOT%scripts\compare_surrogates.py" ^
    --m2 "%ATTR_GPU_NPZ%" ^
    --m3 "%ATTR_GPU_GD_NPZ%" ^
    --top-k 10

echo ========================================
echo  Step 7: Plots (uses GPU/Cholesky attribs)
echo ========================================
python "%ROOT%scripts\explain_instance.py" ^
    --attributions "%ATTR_GPU_NPZ%" ^
    --X "%XBIN%" --preds "%PBIN%" --weights "%WBIN%" ^
    --top-k 15 ^
    --out "%PLOTS%"

echo ========================================
echo  Done.
echo    GPU (Double-Precision) Cholesky attribs : %ATTR_GPU_NPZ%
echo    GPU GD       attribs : %ATTR_GPU_GD_NPZ%
echo    CPU NumPy    attribs : %ATTR_CPU_NPZ%
echo    M2 PyTorch   attribs : %ATTR_M2_NPZ%
echo    Plots                : %PLOTS%\
echo ========================================
