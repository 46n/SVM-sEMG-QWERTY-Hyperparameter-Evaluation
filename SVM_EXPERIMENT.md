# SVM Hyperparameter Experiment

## Project title

Experimental Evaluation of Hyperparameter Variations in an Existing SVM-Based sEMG QWERTY Keypress Classification Model

This SVM-only workspace preserves the original ANSLab-UHN preprocessing and RBF-SVM baseline while providing a Windows-safe, reproducible experiment for the included P1 sample data.

## Anaconda Navigator

The environment is named `semg-svm` and the notebook kernel is **Python (semg-svm SVM)**.

1. Open Anaconda Navigator.
2. Select **Environments**, then select `semg-svm`.
3. Launch JupyterLab.
4. Browse to this repository and open `svm_experiment/SVM_Hyperparameter_Experiment.ipynb`.

Alternatively, double-click `Launch-SVM-Experiment.bat` to open the correct notebook and environment directly.

## Recreate the environment

```powershell
D:\anaconda\Scripts\conda.exe env create --file environment-svm.yml
D:\anaconda\envs\semg-svm\python.exe -m pip install -e . --no-deps
D:\anaconda\envs\semg-svm\python.exe -m ipykernel install --user --name semg-svm --display-name "Python (semg-svm SVM)"
```

The `--no-deps` option is deliberate. The upstream `setup.py` reads the full requirements list, which includes PyTorch/CUDA and W&B packages used by the MLP code but not by this SVM experiment.

## Prepare the included sample

```powershell
D:\anaconda\envs\semg-svm\python.exe -m svm_experiment.prepare_sample_data
```

This applies the authors' validation, 0.2-second window slicing, feature extraction (RMS, LOGVAR, WL, WAMP, AR1, and AR2), and per-day aggregation to P1.

## Run the experiment

```powershell
D:\anaconda\envs\semg-svm\python.exe -m svm_experiment.run_hyperparameter_experiment
```

The default experiment compares the upstream RBF-SVM defaults (`C=1`, `gamma=scale`) with a grid over:

- `C`: 0.1, 1, 10, 100
- `gamma`: scale, auto, 0.001, 0.01, 0.1
- evaluation: within-day stratified holdout and between-day transfer

Results, cross-validation tables, and fitted `joblib` pipelines are written to `outputs/`. The scaler is fitted only on training data through a scikit-learn pipeline, preventing test-data leakage.

The default uses `n_jobs=1` to avoid Windows/Jupyter multiprocessing cleanup warnings. Pass `--n-jobs -1` on the CLI if you prefer to use all CPU cores.

## Data scope and attribution

The GitHub repository contains sample participant P1. The full 19-participant dataset remains available through the link in the upstream README. This fork retains the upstream CC BY-SA 4.0 license and source history.
