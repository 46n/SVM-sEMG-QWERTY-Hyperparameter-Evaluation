@echo off
setlocal
set "PROJECT_DIR=%~dp0"
set "JUPYTER=D:\anaconda\envs\semg-svm\Scripts\jupyter-lab.exe"

if not exist "%JUPYTER%" (
    echo The semg-svm environment was not found.
    echo Recreate it with environment-svm.yml first.
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"
"%JUPYTER%" --notebook-dir="%PROJECT_DIR%" "%PROJECT_DIR%svm_experiment\SVM_Hyperparameter_Experiment.ipynb"
