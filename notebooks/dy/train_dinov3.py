"""Train multitask heads on frozen DINOv3 embeddings for forehead/glabella."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

import train as train_utils
from data_checks import check_train_val_split, embedding_fingerprint
from dataset import ResizeAndPad
from task_config import ClassificationTask, REGRESSION_TARGETS, get_classification_tasks, map_grade


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "trained" / "dy_forehead_glabella"
DEFAULT_TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "dy_forehead_glabella_train_metadata.csv"
DEFAULT_VAL_CSV = PROJECT_ROOT / "data" / "processed" / "dy_forehead_glabella_val_metadata.csv"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_CSV)
    parser.add_argument("--run-name", type=str, default="dy_dinov3_vits16_3grade_head_v1")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--model-revision", type=str, default=None, help="Pin a Hugging Face commit for reproducible features.")
    parser.add_argument("--hf-token", type=str, default="")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--grade-scheme", choices=["original", "three"], default="three")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--embed-batch-size", type=int, default=8)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--dropout-p", type=float, default=0.35)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--ordinal-loss-weight", type=float, default=0.2)
    parser.add_argument("--cls-loss-weight", type=float, default=1.0)
    parser.add_argument("--reg-loss-weight", type=float, default=0.15)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--sampler-power", type=float, default=0.5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    return parser.parse_args()


def resolve_token(explicit_token: str) -> str | None:
    token = explicit_token.strip() if explicit_token else ""
    if token:
        return token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def resolve_checkpoint_dir(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir is not None:
        return args.checkpoint_dir
    return DEFAULT_CHECKPOINT_ROOT / args.run_name


class DinoImageDataset(Dataset):
    """CSV dataset that returns padded PIL images and multitask targets."""

    def __init__(
        self,
        csv_path: Path,
        regression_stats: dict[str, dict[str, float]],
        tasks: tuple[ClassificationTask, ...],
        grade_scheme: str,
        image_size: int,
    ) -> None:
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.regression_stats = regression_stats
        self.tasks = tasks
        self.grade_scheme = grade_scheme
        self.resize_and_pad = ResizeAndPad(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def _classification_targets(self, row: pd.Series) -> torch.Tensor:
        values: list[int] = []
        for task in self.tasks:
            raw_value = row.get(task.column)
            if pd.isna(raw_value) or raw_value == "":
                values.append(-100)
            else:
                values.append(map_grade(int(raw_value), self.grade_scheme))
        return torch.tensor(values, dtype=torch.long)

    def _regression_targets(self, row: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
        values: list[float] = []
        mask: list[bool] = []
        for target in REGRESSION_TARGETS:
            raw_value = row.get(target)
            if pd.isna(raw_value) or raw_value == "":
                values.append(0.0)
                mask.append(False)
                continue
            value = float(raw_value)
            stats = self.regression_stats[target]
            mean = float(stats.get("mean", 0.0))
            std = max(float(stats.get("std", 1.0)), 1e-6)
            values.append((value - mean) / std)
            mask.append(True)
        return (
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.resize_and_pad(image)
        reg_targets, reg_mask = self._regression_targets(row)
        return {
            "image": image,
            "cls_targets": self._classification_targets(row),
            "reg_targets": reg_targets,
            "reg_mask": reg_mask,
            "image_path": str(row["image_path"]),
        }


def collate_image_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [sample["image"] for sample in samples],
        "cls_targets": torch.stack([sample["cls_targets"] for sample in samples]),
        "reg_targets": torch.stack([sample["reg_targets"] for sample in samples]),
        "reg_mask": torch.stack([sample["reg_mask"] for sample in samples]),
        "image_paths": [sample["image_path"] for sample in samples],
    }


def preprocess_images(processor: Any, images: list[Image.Image]) -> torch.Tensor:
    try:
        batch = processor(
            images=images,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
    except TypeError:
        batch = processor(images=images, return_tensors="pt")
    return batch["pixel_values"]


def encoder_forward(encoder: torch.nn.Module, pixel_values: torch.Tensor) -> Any:
    try:
        return encoder(pixel_values=pixel_values, interpolate_pos_encoding=True)
    except TypeError:
        return encoder(pixel_values=pixel_values)


def pooled_features(outputs: Any) -> torch.Tensor:
    pooler_output = getattr(outputs, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output
    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if last_hidden_state is not None:
        return last_hidden_state[:, 0]
    if isinstance(outputs, (tuple, list)) and outputs:
        first = outputs[0]
        if first.ndim == 3:
            return first[:, 0]
        return first
    raise TypeError("Could not find pooled DINOv3 features in model output.")


def cache_matches(
    payload: dict[str, Any], args: argparse.Namespace, csv_path: Path, fingerprint: str
) -> bool:
    meta = payload.get("meta", {})
    if meta.get("model_name") != args.model_name:
        return False
    if int(meta.get("image_size", -1)) != int(args.image_size):
        return False
    if meta.get("grade_scheme") != args.grade_scheme:
        return False
    if str(meta.get("csv_path")) != str(csv_path.resolve()):
        return False
    return meta.get("fingerprint") == fingerprint


def extract_or_load_embeddings(
    csv_path: Path,
    cache_path: Path,
    args: argparse.Namespace,
    regression_stats: dict[str, dict[str, float]],
    tasks: tuple[ClassificationTask, ...],
    processor: Any,
    encoder: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    resolved_revision = getattr(encoder.config, "_commit_hash", None) or args.model_revision
    if resolved_revision is None:
        logger.warning("Encoder revision unavailable: embedding cache will be rebuilt. Pin --model-revision to reuse it.")
    fingerprint = embedding_fingerprint(
        csv_path, regression_stats, processor.to_dict(),
        str(resolved_revision or "unresolved"), args.image_size,
    )
    if cache_path.exists() and not args.force_embeddings:
        payload = torch.load(cache_path, map_location="cpu")
        if resolved_revision is not None and isinstance(payload, dict) and cache_matches(payload, args, csv_path, fingerprint):
            logger.info("loaded embedding cache: %s", cache_path)
            return payload
        logger.info("embedding cache mismatch, rebuilding: %s", cache_path)

    dataset = DinoImageDataset(
        csv_path=csv_path,
        regression_stats=regression_stats,
        tasks=tasks,
        grade_scheme=args.grade_scheme,
        image_size=args.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.embed_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_image_batch,
    )

    features: list[torch.Tensor] = []
    cls_targets: list[torch.Tensor] = []
    reg_targets: list[torch.Tensor] = []
    reg_masks: list[torch.Tensor] = []
    image_paths: list[str] = []

    encoder.eval()
    desc = f"extract {csv_path.stem}"
    with torch.inference_mode():
        for batch in tqdm(loader, desc=desc):
            pixel_values = preprocess_images(processor, batch["images"]).to(device)
            outputs = encoder_forward(encoder, pixel_values)
            features.append(pooled_features(outputs).detach().cpu().float())
            cls_targets.append(batch["cls_targets"])
            reg_targets.append(batch["reg_targets"])
            reg_masks.append(batch["reg_mask"])
            image_paths.extend(batch["image_paths"])

    payload = {
        "features": torch.cat(features, dim=0),
        "cls_targets": torch.cat(cls_targets, dim=0),
        "reg_targets": torch.cat(reg_targets, dim=0),
        "reg_mask": torch.cat(reg_masks, dim=0),
        "image_paths": image_paths,
        "meta": {
            "model_name": args.model_name,
            "image_size": args.image_size,
            "grade_scheme": args.grade_scheme,
            "csv_path": str(csv_path.resolve()),
            "rows": len(dataset),
            "fingerprint": fingerprint,
            "model_revision": resolved_revision,
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    logger.info("saved embedding cache: %s shape=%s", cache_path, tuple(payload["features"].shape))
    return payload


class EmbeddingDataset(Dataset):
    def __init__(self, payload: dict[str, Any]) -> None:
        self.features = payload["features"].float()
        self.cls_targets = payload["cls_targets"].long()
        self.reg_targets = payload["reg_targets"].float()
        self.reg_mask = payload["reg_mask"].bool()

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        features = self.features[idx]
        return {
            "features": features,
            "image": features,
            "cls_targets": self.cls_targets[idx],
            "reg_targets": self.reg_targets[idx],
            "reg_mask": self.reg_mask[idx],
        }


class DinoEmbeddingMultiTask(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        projection_dim: int,
        dropout_p: float,
        tasks: tuple[ClassificationTask, ...],
    ) -> None:
        super().__init__()
        self.tasks = tasks
        self.shared_head = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(projection_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )
        self.classification_heads = nn.ModuleDict(
            {task.name: nn.Linear(projection_dim, task.num_classes) for task in tasks}
        )
        self.regression_head = nn.Linear(projection_dim, len(REGRESSION_TARGETS))

    def forward(self, features: torch.Tensor) -> dict[str, Any]:
        shared = self.shared_head(features)
        cls_outputs = {
            name: head(shared)
            for name, head in self.classification_heads.items()
        }
        reg_outputs = self.regression_head(shared)
        return {"cls": cls_outputs, "reg": reg_outputs}


def build_embedding_loaders(
    train_payload: dict[str, Any],
    val_payload: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader]:
    train_ds = EmbeddingDataset(train_payload)
    val_ds = EmbeddingDataset(val_payload)
    sampler = None
    shuffle = True
    if args.weighted_sampler:
        weights = train_utils.build_sample_weights(
            args.train_csv,
            sampler_power=args.sampler_power,
            grade_scheme=args.grade_scheme,
        )
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False
        logger.info(
            "weighted sampler enabled: min=%.4f max=%.4f mean=%.4f",
            float(weights.min()),
            float(weights.max()),
            float(weights.mean()),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.head_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.head_batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def run_embedding_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: dict[str, torch.Tensor],
    regression_stats: dict[str, dict[str, float]],
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    buffers = train_utils.empty_metric_buffers()

    for batch in loader:
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(batch["features"])
            cls_loss, _ = train_utils.classification_loss(
                outputs=outputs,
                targets=batch["cls_targets"],
                class_weights=class_weights,
                label_smoothing=args.label_smoothing,
                ordinal_loss_weight=args.ordinal_loss_weight,
                focal_gamma=args.focal_gamma,
            )
            reg_loss = train_utils.regression_loss(
                outputs=outputs,
                targets=batch["reg_targets"],
                mask=batch["reg_mask"],
            )
            loss = (args.cls_loss_weight * cls_loss) + (args.reg_loss_weight * reg_loss)

        if training:
            loss.backward()
            if args.grad_clip_norm and args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

        train_utils.update_metric_buffers(buffers, outputs, batch, loss, regression_stats)

    prefix = "train" if training else "val"
    return train_utils.summarize_metrics(buffers, prefix)


def save_dinov3_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_score: float,
    best_epoch: int,
    args: argparse.Namespace,
    regression_stats: dict[str, dict[str, float]],
    metrics: dict[str, float],
    feature_dim: int,
    tasks: tuple[ClassificationTask, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "hf_token"
    }
    payload = {
        "epoch": epoch,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "config": config,
        "feature_dim": feature_dim,
        "encoder_model_name": args.model_name,
        "grade_scheme": args.grade_scheme,
        "classification_tasks": [
            {
                "name": task.name,
                "column": task.column,
                "part_name": task.part_name,
                "num_classes": task.num_classes,
            }
            for task in tasks
        ],
        "regression_targets": list(REGRESSION_TARGETS),
        "regression_stats": regression_stats,
    }
    torch.save(payload, path)


def train(args: argparse.Namespace) -> None:
    tasks = get_classification_tasks(args.grade_scheme)
    train_utils.CLASSIFICATION_TASKS = tasks

    if not args.train_csv.exists():
        raise FileNotFoundError(f"Missing train CSV: {args.train_csv}")
    if not args.val_csv.exists():
        raise FileNotFoundError(f"Missing val CSV: {args.val_csv}")

    split_summary = check_train_val_split(args.train_csv, args.val_csv)
    logger.info("subject-disjoint train/val split: %s", split_summary)

    train_utils.set_seed(args.seed)
    checkpoint_dir = resolve_checkpoint_dir(args)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    if args.max_train_rows > 0:
        args.train_csv = train_utils.subset_csv(
            args.train_csv,
            checkpoint_dir / f"train_subset_{args.max_train_rows}.csv",
            args.max_train_rows,
        )
    if args.max_val_rows > 0:
        args.val_csv = train_utils.subset_csv(
            args.val_csv,
            checkpoint_dir / f"val_subset_{args.max_val_rows}.csv",
            args.max_val_rows,
        )

    token = resolve_token(args.hf_token)
    if token:
        logger.info("Hugging Face token detected.")
    elif not args.local_files_only:
        logger.warning("No HF_TOKEN detected. Gated DINOv3 downloads will fail without login/token.")

    regression_stats = train_utils.build_regression_stats(args.train_csv)
    train_utils.save_regression_stats(regression_stats, checkpoint_dir / "regression_stats.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)
    logger.info("encoder=%s", args.model_name)
    logger.info("checkpoint_dir=%s", checkpoint_dir)

    processor = AutoImageProcessor.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        token=token,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    encoder = AutoModel.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        token=token,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    ).to(device)

    train_payload = extract_or_load_embeddings(
        csv_path=args.train_csv,
        cache_path=checkpoint_dir / "train_embeddings.pt",
        args=args,
        regression_stats=regression_stats,
        tasks=tasks,
        processor=processor,
        encoder=encoder,
        device=device,
    )
    val_payload = extract_or_load_embeddings(
        csv_path=args.val_csv,
        cache_path=checkpoint_dir / "val_embeddings.pt",
        args=args,
        regression_stats=regression_stats,
        tasks=tasks,
        processor=processor,
        encoder=encoder,
        device=device,
    )

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    feature_dim = int(train_payload["features"].shape[1])
    train_loader, val_loader = build_embedding_loaders(train_payload, val_payload, args)
    model = DinoEmbeddingMultiTask(
        feature_dim=feature_dim,
        projection_dim=args.projection_dim,
        dropout_p=args.dropout_p,
        tasks=tasks,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )
    class_weights = train_utils.compute_class_weights(
        args.train_csv,
        device,
        grade_scheme=args.grade_scheme,
        class_weight_power=args.class_weight_power,
    )

    history_path = checkpoint_dir / "history.csv"
    latest_path = checkpoint_dir / "latest.pth"
    best_path = checkpoint_dir / "best.pth"
    best_score = float("-inf")
    best_epoch = -1
    bad_epochs = 0

    logger.info("feature_dim=%d", feature_dim)
    logger.info("classification tasks=%s", [task.name for task in tasks])
    logger.info("regression targets=%s", list(REGRESSION_TARGETS))

    for epoch in range(args.epochs):
        train_metrics = run_embedding_epoch(
            model=model,
            loader=train_loader,
            device=device,
            class_weights=class_weights,
            regression_stats=regression_stats,
            args=args,
            optimizer=optimizer,
        )
        val_metrics = run_embedding_epoch(
            model=model,
            loader=val_loader,
            device=device,
            class_weights=class_weights,
            regression_stats=regression_stats,
            args=args,
            optimizer=None,
        )

        score = val_metrics["val_mean_macro_f1"]
        improved = score > best_score + args.early_stopping_min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        scheduler.step(score)
        lr = float(max(group["lr"] for group in optimizer.param_groups))
        metrics = {**train_metrics, **val_metrics, "lr": lr}
        row = {
            "epoch": epoch + 1,
            **metrics,
            "best_score": best_score,
            "best_epoch": best_epoch + 1 if best_epoch >= 0 else 0,
        }
        train_utils.append_history(history_path, row)
        plot_path = train_utils.save_history_plot(history_path)

        save_dinov3_checkpoint(
            latest_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_score,
            best_epoch,
            args,
            regression_stats,
            metrics,
            feature_dim,
            tasks,
        )
        if improved:
            save_dinov3_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_score,
                best_epoch,
                args,
                regression_stats,
                metrics,
                feature_dim,
                tasks,
            )

        logger.info(
            "epoch %d/%d train_loss=%.4f val_loss=%.4f val_macro_f1=%.4f "
            "val_reg_mae=%.4f lr=%.2e bad_epochs=%d plot=%s",
            epoch + 1,
            args.epochs,
            train_metrics["train_loss"],
            val_metrics["val_loss"],
            score,
            val_metrics["val_mean_reg_mae"],
            lr,
            bad_epochs,
            plot_path,
        )

        if args.early_stopping_patience > 0 and bad_epochs >= args.early_stopping_patience:
            logger.info(
                "early stopping at epoch=%d best_epoch=%d best_score=%.4f",
                epoch + 1,
                best_epoch + 1,
                best_score,
            )
            break

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "latest_checkpoint": str(latest_path),
        "best_checkpoint": str(best_path),
        "history_path": str(history_path),
        "embedding_cache": {
            "train": str(checkpoint_dir / "train_embeddings.pt"),
            "val": str(checkpoint_dir / "val_embeddings.pt"),
        },
        "best_score": best_score,
        "best_epoch": best_epoch + 1 if best_epoch >= 0 else 0,
    }
    summary_path = checkpoint_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("training summary=%s", json.dumps(summary, ensure_ascii=False))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    train(parse_args())


if __name__ == "__main__":
    main()
