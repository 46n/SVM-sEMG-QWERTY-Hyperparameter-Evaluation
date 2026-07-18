"""Prepare the repository's P1 sample data using the upstream feature pipeline.

This wrapper intentionally uses the original validation, windowing, and feature
classes while avoiding two assumptions in the upstream shell workflow:

* only P1 is included in the GitHub sample data; and
* Windows cannot create the upstream log filenames containing colons.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from keypressemg.common.types_defined import DayT1T2, KeyPress, Participant
from keypressemg.data_prep.experiments_read.experiment import Experiment
from keypressemg.data_prep.experiments_read.experiment_reader import ExperimentReader
from keypressemg.data_prep.experiments_read.rhd_data import (
    RHDDataFileLoader,
    RHDDataFileSaver,
)
from keypressemg.data_prep.window_prepare.feature_window import FeatureWindow
from keypressemg.data_prep.window_prepare.window_slice import RHDWindowSlicer


LOGGER = logging.getLogger("prepare_semg_sample")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "CleanData"


def validate_recordings(data_root: Path, participant: Participant) -> Path:
    output_dir = data_root / "valid_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    for day in DayT1T2:
        experiment = Experiment(
            folder_path=data_root,
            participant=participant,
            test=day,
        )
        for key in KeyPress:
            reader = ExperimentReader(experiment=experiment, key=key)
            reader.validate()
            LOGGER.info(
                "%s %s %s: %d validated file(s)",
                participant.value,
                day.value,
                key.value,
                len(reader.validated_files),
            )
            for file_path in reader.validated_files:
                rhd_file = reader.read(file_path)
                RHDDataFileSaver(rhd=rhd_file.rhd_data, save_path=output_dir).save()
    return output_dir


def slice_windows(
    data_root: Path,
    valid_experiments_dir: Path,
    window_length_seconds: float,
) -> Path:
    output_dir = data_root / "valid_windows"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(path for path in valid_experiments_dir.iterdir() if path.is_file())
    for index, file_path in enumerate(files, start=1):
        loader = RHDDataFileLoader(filepath=file_path)
        loader.load()
        slicer = RHDWindowSlicer(
            loader.data,
            win_length_seconds=window_length_seconds,
        )
        slicer.slice()
        slicer.save_windows(output_dir)
        if index % 25 == 0 or index == len(files):
            LOGGER.info("Sliced %d/%d validated recordings", index, len(files))
    return output_dir


def extract_features(data_root: Path, windows_dir: Path) -> Path:
    output_dir = data_root / "valid_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(path for path in windows_dir.iterdir() if path.is_file())
    for index, file_path in enumerate(files, start=1):
        feature_window = FeatureWindow(file_path, feature_windows_path=output_dir)
        feature_window.load_window()
        feature_window.extract_window_features()
        feature_window.save()
        if index % 100 == 0 or index == len(files):
            LOGGER.info("Extracted %d/%d feature windows", index, len(files))
    return output_dir


def aggregate_user_features(
    data_root: Path,
    features_dir: Path,
    participant: Participant,
) -> Path:
    output_dir = data_root / "valid_user_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    for day in DayT1T2:
        prefix = f"{participant.value}_{day.value}"
        arrays: list[np.ndarray] = []
        labels: list[int] = []

        for file_path in sorted(features_dir.glob(f"**/{prefix}*.npy")):
            time_marker = file_path.name.find("_time_")
            if time_marker < 1:
                raise ValueError(f"Cannot infer key label from {file_path.name}")
            key = KeyPress.from_str(file_path.name[time_marker - 1].lower())
            if key is KeyPress.SPACE:
                continue
            with file_path.open("rb") as feature_file:
                arrays.append(np.load(feature_file).reshape(1, -1))
            labels.append(key.to_num())

        if not arrays:
            raise FileNotFoundError(f"No features found for {prefix} in {features_dir}")

        X = np.concatenate(arrays, axis=0)
        y = np.asarray(labels, dtype=np.int64)
        np.save(output_dir / f"{prefix}_X.npy", X)
        np.save(output_dir / f"{prefix}_y.npy", y)
        LOGGER.info("Saved %s: X%s y%s", prefix, X.shape, y.shape)
    return output_dir


def prepare_pipeline(
    data_root: Path = DEFAULT_DATA_ROOT,
    participant: Participant = Participant.P1,
    window_length_seconds: float = 0.2,
) -> Path:
    data_root = data_root.resolve()
    participant_dir = data_root / participant.value
    if not participant_dir.is_dir():
        raise FileNotFoundError(f"Sample directory not found: {participant_dir}")

    LOGGER.info("Preparing %s from %s", participant.value, data_root)
    validated_dir = validate_recordings(data_root, participant)
    windows_dir = slice_windows(data_root, validated_dir, window_length_seconds)
    features_dir = extract_features(data_root, windows_dir)
    return aggregate_user_features(data_root, features_dir, participant)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--participant",
        choices=[participant.value for participant in Participant],
        default=Participant.P1.value,
    )
    parser.add_argument("--window-length-seconds", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    output_dir = prepare_pipeline(
        data_root=args.data_root,
        participant=Participant(args.participant),
        window_length_seconds=args.window_length_seconds,
    )
    print(f"Prepared feature arrays: {output_dir}")


if __name__ == "__main__":
    main()
