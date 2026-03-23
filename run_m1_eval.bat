@echo off
REM ======================================================
REM  Milestone 1 Full Evaluation Script (Windows)
REM  Runs GPU + CPU sweeps, validates, combines timings,
REM  and generates comparison plots.
REM ======================================================
setlocal enabledelayedexpansion

set ROOT=%~dp0
set BUILD=%ROOT%build
set EXE=%BUILD%\Release\lime_m1.exe
set GPU_TXT=%BUILD%\timings_gpu.txt
set CPU_TXT=%BUILD%\timings_cpu.txt
set CSV=%BUILD%\timings_m1.csv

REM --- Configuration ---
set D_VALUES=128 256 512
set B_VALUES=512 1024 2048 4096 8192 16384
set REPEATS=6

REM Clean old timing files
if exist "%GPU_TXT%" del "%GPU_TXT%"
if exist "%CPU_TXT%" del "%CPU_TXT%"

echo ========================================
echo  Step 1: Run GPU sweeps (full pipeline)
echo ========================================
for %%D in (%D_VALUES%) do (
    for %%B in (%B_VALUES%) do (
        for /l %%R in (1,1,%REPEATS%) do (
            echo [GPU] D=%%D B=%%B run=%%R/%REPEATS% ...
            "%EXE%" --D=%%D --B=%%B > "%BUILD%\_tmp_gpu.txt" 2>&1
            REM Parse timing line and append with D,B,run prefix
            for /f "tokens=*" %%L in ('findstr /B "Timing" "%BUILD%\_tmp_gpu.txt"') do (
                echo D=%%D B=%%B run=%%R %%L >> "%GPU_TXT%"
            )
        )
    )
)

echo ========================================
echo  Step 2: Run CPU sweeps (full pipeline)
echo ========================================
for %%D in (%D_VALUES%) do (
    for %%B in (%B_VALUES%) do (
        for /l %%R in (1,1,%REPEATS%) do (
            echo [CPU] D=%%D B=%%B run=%%R/%REPEATS% ...
            python "%ROOT%scripts\cpu_reference.py" --B %%B --D %%D > "%BUILD%\_tmp_cpu.txt" 2>&1
            for /f "tokens=*" %%L in ('findstr /B "Timing" "%BUILD%\_tmp_cpu.txt"') do (
                echo D=%%D B=%%B run=%%R %%L >> "%CPU_TXT%"
            )
        )
    )
)

echo ========================================
echo  Step 3: Convert timings to CSV
echo ========================================
python "%ROOT%scripts\convert_timings_sweep.py" --gpu "%GPU_TXT%" --cpu "%CPU_TXT%" --drop-first 1 --out "%CSV%"

echo ========================================
echo  Step 4: Validate (D=128, B=1024)
echo ========================================
set XVAL=%BUILD%\X_D128_B1024.bin
set ZVAL=%BUILD%\zprime_D128_B1024.bin
echo Running GPU for validation ...
"%EXE%" --D=128 --B=1024 --write-X "%XVAL%" --write-zprime "%ZVAL%"
echo Running comparison ...
python "%ROOT%scripts\compare_and_validate.py" --X "%XVAL%" --zprime "%ZVAL%" --B 1024 --D 128 --preds preds.bin --weights weights.bin --tol_abs 1e-3 --tol_rel 1e-2

echo ========================================
echo  Step 5: Generate plots
echo ========================================
python "%ROOT%scripts\plot_results.py" --csv "%CSV%" --out "%ROOT%plots"

echo ========================================
echo  Done! Results:
echo    CSV:   %CSV%
echo    Plots: %ROOT%plots\
echo ========================================
