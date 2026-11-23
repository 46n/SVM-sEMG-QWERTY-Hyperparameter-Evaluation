from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from svm_experiment.run_hyperparameter_experiment import Scenario, evaluate_scenario


class SVMExperimentTests(unittest.TestCase):
    def test_small_grid_produces_baseline_and_tuned_results(self) -> None:
        rng = np.random.default_rng(42)
        X = np.vstack(
            [rng.normal(loc=class_id * 2.0, scale=0.25, size=(12, 6)) for class_id in range(3)]
        )
        y = np.repeat(np.arange(3), 12)
        train_indices = np.concatenate([np.arange(class_id * 12, class_id * 12 + 10) for class_id in range(3)])
        test_indices = np.concatenate([np.arange(class_id * 12 + 10, class_id * 12 + 12) for class_id in range(3)])
        scenario = Scenario(
            name="synthetic",
            mode="split-day",
            train_day="T1",
            test_day="T1",
            X_train=X[train_indices],
            y_train=y[train_indices],
            X_test=X[test_indices],
            y_test=y[test_indices],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            rows = evaluate_scenario(
                scenario,
                C_values=(1.0,),
                gamma_values=("scale",),
                cv_folds=2,
                random_state=42,
                n_jobs=1,
                output_dir=Path(temp_dir),
            )

        self.assertEqual([row["variant"] for row in rows], ["upstream_default", "tuned"])


if __name__ == "__main__":
    unittest.main()
