"""Checks shared by the forehead/glabella training entry points."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def check_train_val_split(train_csv: Path, val_csv: Path) -> dict[str, int]:
    """Stop if the same subject or source image appears on both sides of the split."""
    train = pd.read_csv(train_csv, dtype={"id": "string"})
    val = pd.read_csv(val_csv, dtype={"id": "string"})
    required = {"id", "image_path"}
    for path, frame in ((train_csv, train), (val_csv, val)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: required columns missing: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{path}: metadata has no rows")
        ids = frame["id"].astype("string").str.strip()
        if ids.isna().any() or ids.eq("").any() or ids.str.lower().eq("unknown").any():
            raise ValueError(f"{path}: empty or unknown subject ID; cannot verify the split")
        images = frame["image_path"].astype("string").str.strip()
        if images.isna().any() or images.eq("").any():
            raise ValueError(f"{path}: missing crop image path")

    train_ids = set(train["id"].astype("string").str.strip())
    val_ids = set(val["id"].astype("string").str.strip())
    shared_ids = train_ids & val_ids

    # Crops from the same original photo have different output paths. Compare
    # source paths when available, as well as the cropped image paths.
    shared_images = set(train["image_path"].astype(str)) & set(val["image_path"].astype(str))
    if "original_image_path" in train and "original_image_path" in val:
        train_sources = set(train["original_image_path"].dropna().astype(str).str.strip()) - {""}
        val_sources = set(val["original_image_path"].dropna().astype(str).str.strip()) - {""}
        shared_images |= train_sources & val_sources
    if shared_ids or shared_images:
        raise ValueError(
            "Train/val leakage detected: "
            f"{len(shared_ids)} shared subject IDs, {len(shared_images)} shared source images. "
            "Inspect the metadata and create a subject-disjoint split before training."
        )
    return {
        "train_rows": len(train), "val_rows": len(val),
        "train_subjects": len(train_ids), "val_subjects": len(val_ids),
    }


def embedding_fingerprint(
    csv_path: Path,
    regression_stats: dict[str, Any],
    processor_config: dict[str, Any],
    model_revision: str,
    image_size: int,
) -> str:
    """Hash labels, preprocessing and image file metadata for a feature cache."""
    digest = hashlib.sha256()
    with csv_path.open("rb") as csv_file:
        for chunk in iter(lambda: csv_file.read(1024 * 1024), b""):
            digest.update(chunk)
    settings = {
        "regression_stats": regression_stats,
        "processor_config": processor_config,
        "model_revision": model_revision,
        "image_size": image_size,
    }
    digest.update(json.dumps(settings, sort_keys=True, default=str).encode("utf-8"))
    for raw_path in pd.read_csv(csv_path, usecols=["image_path"])["image_path"]:
        path = Path(str(raw_path))
        stat = path.stat()  # Missing input must fail, even if a cache exists.
        digest.update(f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check subject and image overlap before model training.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check_train_val_split(args.train_csv, args.val_csv), indent=2))


if __name__ == "__main__":
    main()
