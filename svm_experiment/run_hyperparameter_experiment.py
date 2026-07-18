"""Evaluate RBF-SVM hyperparameters on the sEMG QWERTY feature vectors."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "CleanData" / "valid_user_features"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class Scenario:
    name: str
    mode: str
    train_day: str
    test_day: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_day(data_dir: Path, participant: str, day: str) -> tuple[np.ndarray, np.ndarray]:
    X_path = data_dir / f"{participant}_{day}_X.npy"
    y_path = data_dir / f"{participant}_{day}_y.npy"
    if not X_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(
            f"Missing prepared arrays for {participant} {day}. "
            "Run: python -m svm_experiment.prepare_sample_data"
        )
    X = np.load(X_path)
    y = np.load(y_path).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Sample/label mismatch for {participant} {day}: {X.shape} vs {y.shape}")
    return X, y


def make_pipeline(C: float = 1.0, gamma: str | float = "scale") -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=C, gamma=gamma)),
        ]
    )


def metric_row(
    scenario: Scenario,
    variant: str,
    model: Pipeline,
    best_params: dict[str, object],
) -> dict[str, object]:
    predictions = model.predict(scenario.X_test)
    return {
        "scenario": scenario.name,
        "mode": scenario.mode,
        "train_day": scenario.train_day,
        "test_day": scenario.test_day,
        "variant": variant,
        "train_samples": len(scenario.y_train),
        "test_samples": len(scenario.y_test),
        "accuracy": accuracy_score(scenario.y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(scenario.y_test, predictions),
        "f1_macro": f1_score(scenario.y_test, predictions, average="macro", zero_division=0),
        "best_params": json.dumps(best_params, sort_keys=True),
    }


def evaluate_scenario(
    scenario: Scenario,
    C_values: Sequence[float],
    gamma_values: Sequence[str | float],
    cv_folds: int,
    random_state: int,
    n_jobs: int,
    output_dir: Path,
) -> list[dict[str, object]]:
    baseline = make_pipeline()
    baseline.fit(scenario.X_train, scenario.y_train)
    rows = [
        metric_row(
            scenario,
            variant="upstream_default",
            model=baseline,
            best_params={"svc__C": 1.0, "svc__gamma": "scale", "svc__kernel": "rbf"},
        )
    ]

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        estimator=make_pipeline(),
        param_grid={"svc__C": list(C_values), "svc__gamma": list(gamma_values)},
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=True,
        refit=True,
    )
    grid.fit(scenario.X_train, scenario.y_train)
    rows.append(
        metric_row(
            scenario,
            variant="tuned",
            model=grid.best_estimator_,
            best_params={**grid.best_params_, "svc__kernel": "rbf"},
        )
    )

    safe_name = scenario.name.replace(" ", "_").replace("/", "-")
    pd.DataFrame(grid.cv_results_).to_csv(output_dir / f"grid_{safe_name}.csv", index=False)
    joblib.dump(grid.best_estimator_, output_dir / f"model_{safe_name}.joblib")
    return rows


def build_scenarios(
    data_dir: Path,
    participant: str,
    mode: str,
    test_size: float,
    random_state: int,
) -> list[Scenario]:
    T1_X, T1_y = load_day(data_dir, participant, "T1")
    T2_X, T2_y = load_day(data_dir, participant, "T2")
    scenarios: list[Scenario] = []

    if mode in {"split-day", "both"}:
        for day, X, y in (("T1", T1_X, T1_y), ("T2", T2_X, T2_y)):
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
            scenarios.append(
                Scenario(
                    name=f"{participant}_{day}_split_day",
                    mode="split-day",
                    train_day=day,
                    test_day=day,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                )
            )

    if mode in {"between-days", "both"}:
        scenarios.extend(
            [
                Scenario(
                    name=f"{participant}_T1_to_T2",
                    mode="between-days",
                    train_day="T1",
                    test_day="T2",
                    X_train=T1_X,
                    y_train=T1_y,
                    X_test=T2_X,
                    y_test=T2_y,
                ),
                Scenario(
                    name=f"{participant}_T2_to_T1",
                    mode="between-days",
                    train_day="T2",
                    test_day="T1",
                    X_train=T2_X,
                    y_train=T2_y,
                    X_test=T1_X,
                    y_test=T1_y,
                ),
            ]
        )
    return scenarios


def run_experiments(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    participant: str = "P1",
    mode: str = "both",
    C_values: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    gamma_values: Sequence[str | float] = ("scale", "auto", 0.001, 0.01, 0.1),
    cv_folds: int = 3,
    test_size: float = 0.2,
    random_state: int = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = build_scenarios(data_dir, participant, mode, test_size, random_state)
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        print(f"Evaluating {scenario.name}...")
        rows.extend(
            evaluate_scenario(
                scenario,
                C_values=C_values,
                gamma_values=gamma_values,
                cv_folds=cv_folds,
                random_state=random_state,
                n_jobs=n_jobs,
                output_dir=output_dir,
            )
        )

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "svm_experiment_summary.csv", index=False)
    (output_dir / "svm_experiment_summary.json").write_text(
        results.to_json(orient="records", indent=2),
        encoding="utf-8",
    )
    return results


def parse_gamma_values(text: str) -> list[str | float]:
    values: list[str | float] = []
    for value in text.split(","):
        stripped = value.strip().lower()
        values.append(stripped if stripped in {"scale", "auto"} else float(stripped))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--participant", default="P1")
    parser.add_argument("--mode", choices=["split-day", "between-days", "both"], default="both")
    parser.add_argument("--C-values", default="0.1,1,10,100")
    parser.add_argument("--gamma-values", default="scale,auto,0.001,0.01,0.1")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_experiments(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        participant=args.participant,
        mode=args.mode,
        C_values=[float(value) for value in args.C_values.split(",")],
        gamma_values=parse_gamma_values(args.gamma_values),
        cv_folds=args.cv_folds,
        test_size=args.test_size,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    print(results.to_string(index=False))
    print(f"\nSaved outputs to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
