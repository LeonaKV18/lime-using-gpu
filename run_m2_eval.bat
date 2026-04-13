@echo off
REM ======================================================
REM  Milestone 2 Evaluation Script
REM  Runs the full pipeline: model training -> M1 GPU ->
REM  M1 validation -> M2 surrogate training -> explanation
REM ======================================================
setlocal enabledelayedexpansion

set ROOT=%~dp0
set BUILD=%ROOT%build
set EXE=%BUILD%\Release\lime_m1.exe
set MODELS=%ROOT%models
set PLOTS=%ROOT%plots\m2

set DATASET=breast_cancer
set MODEL_BIN=%MODELS%\%DATASET%.bin
set MODEL_NPZ=%MODELS%\%DATASET%.npz

set B=16384
set XBIN=%BUILD%\X_m2.bin
set PBIN=%BUILD%\preds_m2.bin
set WBIN=%BUILD%\weights_m2.bin
set ZBIN=%BUILD%\zprime_m2.bin
set ATTR=%BUILD%\attributions.npz

echo ========================================
echo  Step 1: Train logistic regression model
echo ========================================
if not exist "%MODEL_BIN%" (
    python "%ROOT%scripts\train_model.py" --dataset=%DATASET% --out-dir="%MODELS%" --instance=30
) else (
    echo Model already present at %MODEL_BIN%, skipping training.
)

echo ========================================
echo  Step 2: Run M1 GPU pipeline
echo ========================================
cd /d "%BUILD%"
"%EXE%" --B=%B% --model="%MODEL_BIN%" ^
    --write-X "%XBIN%" --write-zprime "%ZBIN%" ^
    --write-preds "%PBIN%" --write-weights "%WBIN%"
cd /d "%ROOT%"

echo ========================================
echo  Step 3: Validate M1 GPU outputs
echo ========================================
python "%ROOT%scripts\compare_and_validate.py" ^
    --X "%XBIN%" --zprime "%ZBIN%" ^
    --B %B% --model "%MODEL_NPZ%" ^
    --preds "%PBIN%" --weights "%WBIN%" ^
    --tol_abs 1e-3 --tol_rel 1e-2

echo ========================================
echo  Step 4: Train surrogate model (M2)
echo ========================================
set KMP_DUPLICATE_LIB_OK=TRUE
python "%ROOT%scripts\surrogate_train.py" ^
    --X "%XBIN%" --preds "%PBIN%" --weights "%WBIN%" ^
    --model "%MODEL_NPZ%" ^
    --B %B% --epochs 500 --lr 0.01 ^
    --out "%ATTR%"

echo ========================================
echo  Step 5: Generate explanations and plots
echo ========================================
python "%ROOT%scripts\explain_instance.py" ^
    --attributions "%ATTR%" ^
    --X "%XBIN%" --preds "%PBIN%" --weights "%WBIN%" ^
    --top-k 15 ^
    --out "%PLOTS%"

echo ========================================
echo  Done.
echo    Attributions : %ATTR%
echo    Plots        : %PLOTS%\
echo ========================================