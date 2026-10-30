"""Small, data-free checks for split integrity and cache invalidation."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_checks import check_train_val_split, embedding_fingerprint


class DataChecksTest(unittest.TestCase):
    def test_same_subject_in_different_images_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, val = root / "train.csv", root / "val.csv"
            pd.DataFrame([{"id": "0007", "image_path": "first.jpg"}]).to_csv(train, index=False)
            pd.DataFrame([{"id": "0007", "image_path": "second.jpg"}]).to_csv(val, index=False)
            with self.assertRaisesRegex(ValueError, "shared subject IDs"):
                check_train_val_split(train, val)

    def test_distinct_subjects_pass_with_zero_padded_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, val = root / "train.csv", root / "val.csv"
            pd.DataFrame([{"id": "0007", "image_path": "first.jpg"}]).to_csv(train, index=False)
            pd.DataFrame([{"id": "0008", "image_path": "second.jpg"}]).to_csv(val, index=False)
            self.assertEqual(check_train_val_split(train, val)["train_subjects"], 1)

    def test_same_source_photo_with_different_crop_paths_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, val = root / "train.csv", root / "val.csv"
            pd.DataFrame([{"id": "0007", "image_path": "crop_a.jpg", "original_image_path": "source.jpg"}]).to_csv(train, index=False)
            pd.DataFrame([{"id": "0008", "image_path": "crop_b.jpg", "original_image_path": "source.jpg"}]).to_csv(val, index=False)
            with self.assertRaisesRegex(ValueError, "shared source images"):
                check_train_val_split(train, val)

    def test_cache_fingerprint_changes_when_label_or_image_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, csv = root / "crop.jpg", root / "metadata.csv"
            image.write_bytes(b"one")
            pd.DataFrame([{"image_path": str(image), "label": 0}]).to_csv(csv, index=False)
            def fingerprint():
                return embedding_fingerprint(csv, {"mean": 1}, {"size": 224}, "commit-1", 224)
            initial = fingerprint()
            pd.DataFrame([{"image_path": str(image), "label": 1}]).to_csv(csv, index=False)
            labels_changed = fingerprint()
            self.assertNotEqual(initial, labels_changed)
            image.write_bytes(b"a longer replacement image")
            self.assertNotEqual(labels_changed, fingerprint())


if __name__ == "__main__":
    unittest.main()
