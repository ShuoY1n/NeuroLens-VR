"""Resolve OASIS-TRT T1 + DKT31_CMA label paths for a subject id."""

from __future__ import annotations

from pathlib import Path

from config import DATASET_ROOT


def subject_folder_name(subject_id: str) -> str:
    return f"OASIS-TRT-20-{subject_id.strip()}"


def t1_path(subject_folder: str) -> Path:
    return (
        DATASET_ROOT
        / "OASIS-TRT-20_volumes"
        / subject_folder
        / "t1weighted_brain.MNI152.nii.gz"
    )


def labels_path(subject_id: str) -> Path:
    sid = subject_id.strip()
    return (
        DATASET_ROOT
        / "OASIS-TRT-20_DKT31_CMA_labels_in_MNI152_v2"
        / f"OASIS-TRT-20-{sid}_DKT31_CMA_labels_in_MNI152.nii.gz"
    )


def require_existing_paths(subject_id: str) -> tuple[Path, Path, str]:
    """Return (t1_nii, labels_nii, subject_folder) or raise SystemExit."""
    subject_folder = subject_folder_name(subject_id)
    t1 = t1_path(subject_folder)
    labels = labels_path(subject_id)
    if not t1.is_file():
        raise SystemExit(f"Missing T1 volume: {t1}")
    if not labels.is_file():
        raise SystemExit(f"Missing label volume: {labels}")
    return t1, labels, subject_folder
