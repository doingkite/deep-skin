"""Train a ResNet-50 multitask model for forehead/glabella skin analysis."""

from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from datetime import datetime
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from data_checks import check_train_val_split
from dataset import ForeheadGlabellaDataset, get_transforms
from model import DEFAULT_LOCAL_RESNET50, get_model
from task_config import CLASSIFICATION_TASKS, REGRESSION_TARGETS, get_classification_tasks, map_grade


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "trained" / "dy_forehead_glabella"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "dy_forehead_glabella_train_metadata.csv",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "dy_forehead_glabella_val_metadata.csv",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--backbone-lr-mult",
        type=float,
        default=0.1,
        help="Use a smaller LR for the pretrained ResNet backbone than for task heads.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout-p", type=float, default=0.30)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument(
        "--grade-scheme",
        choices=["original", "three"],
        default="original",
        help="Use original labels or grouped low/middle/high 3-class labels.",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=0.0,
        help="Optional focal-loss gamma for imbalanced classes. 0 keeps standard cross entropy.",
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=1.0,
        help="Apply class weights as weight ** power. Lower values reduce rare-class overcorrection.",
    )
    parser.add_argument("--cls-loss-weight", type=float, default=1.0)
    parser.add_argument("--reg-loss-weight", type=float, default=0.35)
    parser.add_argument(
        "--ordinal-loss-weight",
        type=float,
        default=0.0,
        help="Add a small order-aware loss to penalize farther grade mistakes more strongly.",
    )
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--flip-prob", type=float, default=0.25)
    parser.add_argument("--jitter-brightness", type=float, default=0.18)
    parser.add_argument("--jitter-contrast", type=float, default=0.18)
    parser.add_argument("--jitter-saturation", type=float, default=0.08)
    parser.add_argument("--affine-prob", type=float, default=0.35)
    parser.add_argument("--affine-degrees", type=float, default=4.0)
    parser.add_argument("--affine-translate", type=float, default=0.015)
    parser.add_argument("--affine-scale-min", type=float, default=0.98)
    parser.add_argument("--affine-scale-max", type=float, default=1.02)
    parser.add_argument("--blur-prob", type=float, default=0.12)
    parser.add_argument("--erasing-prob", type=float, default=0.10)
    parser.add_argument(
        "--weighted-sampler",
        action="store_true",
        help="Oversample rare grade rows. Repeated minority rows receive fresh random augmentation.",
    )
    parser.add_argument(
        "--sampler-power",
        type=float,
        default=0.5,
        help="Class-frequency inverse power for weighted sampler. 0.5 is gentler than full inverse frequency.",
    )
    parser.add_argument(
        "--weights",
        choices=["local_or_none", "local", "torchvision", "none"],
        default="local",
        help="ResNet-50 initialization. Default requires local pretrained weights; torchvision may download.",
    )
    parser.add_argument("--pretrained-path", type=Path, default=DEFAULT_LOCAL_RESNET50)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--freeze-until",
        choices=["none", "stem", "layer1", "layer2", "layer3"],
        default="none",
        help="Freeze early ResNet stages. Use layer2/layer3 to reduce overfitting.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Use only the first N train rows for a quick smoke/diagnostic run.",
    )
    parser.add_argument(
        "--max-val-rows",
        type=int,
        default=0,
        help="Use only the first N validation rows for a quick smoke/diagnostic run.",
    )
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Clip gradients to stabilize fine-tuning. Set 0 to disable.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _path_to_str(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def build_regression_stats(train_csv: Path) -> dict[str, dict[str, float]]:
    df = pd.read_csv(train_csv)
    stats: dict[str, dict[str, float]] = {}
    for target in REGRESSION_TARGETS:
        if target not in df.columns:
            stats[target] = {"mean": 0.0, "std": 1.0, "count": 0}
            continue
        values = pd.to_numeric(df[target], errors="coerce").dropna()
        if values.empty:
            stats[target] = {"mean": 0.0, "std": 1.0, "count": 0}
            continue
        std = float(values.std(ddof=0))
        stats[target] = {
            "mean": float(values.mean()),
            "std": std if std > 1e-6 else 1.0,
            "count": int(values.shape[0]),
        }
    return stats


def save_regression_stats(stats: dict[str, dict[str, float]], path: Path) -> None:
    payload = {"targets": stats, "target_order": list(REGRESSION_TARGETS)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def subset_csv(source: Path, destination: Path, max_rows: int) -> Path:
    if max_rows <= 0:
        return source
    df = pd.read_csv(source).head(max_rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False, encoding="utf-8-sig")
    logger.info("subset csv saved: %s (%d rows from %s)", destination, len(df), source)
    return destination


def build_dataloaders(
    args: argparse.Namespace,
    regression_stats: dict[str, dict[str, float]],
) -> tuple[DataLoader, DataLoader]:
    train_ds = ForeheadGlabellaDataset(
        csv_path=args.train_csv,
        transform=get_transforms(
            image_size=args.image_size,
            split="train",
            flip_prob=args.flip_prob,
            jitter_brightness=args.jitter_brightness,
            jitter_contrast=args.jitter_contrast,
            jitter_saturation=args.jitter_saturation,
            affine_prob=args.affine_prob,
            affine_degrees=args.affine_degrees,
            affine_translate=args.affine_translate,
            affine_scale_min=args.affine_scale_min,
            affine_scale_max=args.affine_scale_max,
            blur_prob=args.blur_prob,
            erasing_prob=args.erasing_prob,
        ),
        regression_stats=regression_stats,
        tasks=CLASSIFICATION_TASKS,
        grade_scheme=args.grade_scheme,
    )
    val_ds = ForeheadGlabellaDataset(
        csv_path=args.val_csv,
        transform=get_transforms(image_size=args.image_size, split="val"),
        regression_stats=regression_stats,
        tasks=CLASSIFICATION_TASKS,
        grade_scheme=args.grade_scheme,
    )
    sampler = None
    shuffle = True
    if args.weighted_sampler:
        sample_weights = build_sample_weights(
            args.train_csv,
            sampler_power=args.sampler_power,
            grade_scheme=args.grade_scheme,
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
        logger.info(
            "weighted sampler enabled: min=%.4f max=%.4f mean=%.4f",
            float(sample_weights.min()),
            float(sample_weights.max()),
            float(sample_weights.mean()),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def build_sample_weights(
    csv_path: Path,
    sampler_power: float = 0.5,
    grade_scheme: str = "original",
) -> torch.Tensor:
    """Build per-row sampler weights from available multitask labels."""
    df = pd.read_csv(csv_path)
    row_weights = np.ones(len(df), dtype=np.float64)

    for task in CLASSIFICATION_TASKS:
        labels = pd.to_numeric(df.get(task.column), errors="coerce")
        valid = labels.notna()
        if not valid.any():
            continue

        mapped_labels = pd.Series(np.nan, index=df.index, dtype="float64")
        mapped_labels.loc[valid] = labels.loc[valid].astype(int).map(
            lambda grade: map_grade(int(grade), grade_scheme)
        )
        counts = mapped_labels.loc[valid].astype(int).value_counts().to_dict()
        task_weights = np.ones(len(df), dtype=np.float64)
        for label, count in counts.items():
            if count > 0:
                task_weights[mapped_labels == label] = float(count) ** (-sampler_power)

        positive = task_weights[valid]
        if positive.size > 0 and positive.mean() > 0:
            task_weights[valid] = positive / positive.mean()
        row_weights = np.maximum(row_weights, task_weights)

    row_weights = row_weights / max(row_weights.mean(), 1e-8)
    return torch.tensor(row_weights, dtype=torch.double)


def compute_class_weights(
    csv_path: Path,
    device: torch.device,
    class_weight_power: float = 1.0,
    grade_scheme: str = "original",
) -> dict[str, torch.Tensor]:
    df = pd.read_csv(csv_path)
    weights: dict[str, torch.Tensor] = {}
    for idx, task in enumerate(CLASSIFICATION_TASKS):
        values = (
            pd.to_numeric(df.get(task.column), errors="coerce")
            .dropna()
            .astype(int)
            .map(lambda grade: map_grade(int(grade), grade_scheme))
        )
        counts = values.value_counts().reindex(range(task.num_classes), fill_value=0).sort_index()
        total = int(counts.sum())
        raw_weights: list[float] = []
        for count in counts.tolist():
            raw_weights.append(total / (task.num_classes * count) if count > 0 else 0.0)
        tensor = torch.tensor(raw_weights, dtype=torch.float32, device=device)
        positive = tensor > 0
        if positive.any():
            tensor[positive] = tensor[positive] / tensor[positive].mean()
            if class_weight_power != 1.0:
                tensor[positive] = tensor[positive].pow(float(class_weight_power))
                tensor[positive] = tensor[positive] / tensor[positive].mean()
        weights[task.name] = tensor
        logger.info("class weights [%s/%d]=%s", task.name, idx, [round(float(v), 4) for v in tensor.cpu()])
    return weights


def classification_loss(
    outputs: dict[str, Any],
    targets: torch.Tensor,
    class_weights: dict[str, torch.Tensor],
    label_smoothing: float,
    ordinal_loss_weight: float,
    focal_gamma: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    total = targets.new_tensor(0.0, dtype=torch.float32)
    pieces: dict[str, float] = {}
    active_tasks = 0
    cls_outputs: dict[str, torch.Tensor] = outputs["cls"]

    for task_idx, task in enumerate(CLASSIFICATION_TASKS):
        task_targets = targets[:, task_idx]
        mask = task_targets >= 0
        if not mask.any():
            pieces[f"loss_{task.name}"] = 0.0
            continue
        logits = cls_outputs[task.name][mask]
        active_targets = task_targets[mask]
        ce_loss = F.cross_entropy(
            logits,
            active_targets,
            weight=class_weights.get(task.name),
            label_smoothing=label_smoothing,
            reduction="none",
        )
        if focal_gamma and focal_gamma > 0:
            probs = F.softmax(logits, dim=1)
            pt = probs.gather(1, active_targets.view(-1, 1)).squeeze(1).clamp_min(1e-6)
            ce_loss = ((1.0 - pt) ** focal_gamma) * ce_loss
        loss = ce_loss.mean()
        if ordinal_loss_weight > 0:
            class_indices = torch.arange(logits.size(1), device=logits.device, dtype=logits.dtype)
            expected_grade = (F.softmax(logits, dim=1) * class_indices).sum(dim=1)
            denom = float(max(task.num_classes - 1, 1))
            ordinal_loss = F.smooth_l1_loss(
                expected_grade / denom,
                active_targets.to(dtype=logits.dtype) / denom,
            )
            loss = loss + (ordinal_loss_weight * ordinal_loss)
            pieces[f"loss_{task.name}_ordinal"] = float(ordinal_loss.detach().cpu())
        total = total + loss
        pieces[f"loss_{task.name}"] = float(loss.detach().cpu())
        active_tasks += 1

    if active_tasks > 0:
        total = total / active_tasks
    return total, pieces


def regression_loss(
    outputs: dict[str, Any],
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pred = outputs["reg"]
    if not mask.any():
        return pred.new_tensor(0.0)
    loss = F.smooth_l1_loss(pred, targets, reduction="none")
    return loss[mask].mean()


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    backbone_lr_mult: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    backbone_params: list[torch.nn.Parameter] = []
    head_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups: list[dict[str, Any]] = []
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": learning_rate * backbone_lr_mult,
                "weight_decay": weight_decay,
            }
        )
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        )
    if not param_groups:
        raise ValueError("No trainable parameters found.")
    return torch.optim.AdamW(param_groups)


def make_grad_scaler(use_amp: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def autocast_context(use_amp: bool) -> Any:
    if not use_amp:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda")
    return torch.cuda.amp.autocast()


def macro_f1_score(y_true: list[int], y_pred: list[int], num_classes: int) -> float:
    if not y_true:
        return 0.0
    scores: list[float] = []
    for label in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        denom = (2 * tp) + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return float(sum(scores) / len(scores))


def empty_metric_buffers() -> dict[str, Any]:
    return {
        "loss_sum": 0.0,
        "samples": 0,
        "cls_true": {task.name: [] for task in CLASSIFICATION_TASKS},
        "cls_pred": {task.name: [] for task in CLASSIFICATION_TASKS},
        "reg_abs_error_sum": np.zeros(len(REGRESSION_TARGETS), dtype=np.float64),
        "reg_counts": np.zeros(len(REGRESSION_TARGETS), dtype=np.int64),
    }


def update_metric_buffers(
    buffers: dict[str, Any],
    outputs: dict[str, Any],
    batch: dict[str, torch.Tensor],
    loss: torch.Tensor,
    regression_stats: dict[str, dict[str, float]],
) -> None:
    images = batch["image"]
    buffers["loss_sum"] += float(loss.detach().cpu()) * images.size(0)
    buffers["samples"] += images.size(0)

    cls_outputs: dict[str, torch.Tensor] = outputs["cls"]
    cls_targets = batch["cls_targets"]
    for task_idx, task in enumerate(CLASSIFICATION_TASKS):
        targets = cls_targets[:, task_idx]
        mask = targets >= 0
        if not mask.any():
            continue
        preds = cls_outputs[task.name].argmax(dim=1)
        buffers["cls_true"][task.name].extend(targets[mask].detach().cpu().tolist())
        buffers["cls_pred"][task.name].extend(preds[mask].detach().cpu().tolist())

    reg_pred = outputs["reg"].detach().cpu()
    reg_target = batch["reg_targets"].detach().cpu()
    reg_mask = batch["reg_mask"].detach().cpu()
    for idx, target_name in enumerate(REGRESSION_TARGETS):
        mask = reg_mask[:, idx]
        if not mask.any():
            continue
        stats = regression_stats[target_name]
        std = max(float(stats.get("std", 1.0)), 1e-6)
        pred_raw = reg_pred[mask, idx] * std + float(stats.get("mean", 0.0))
        target_raw = reg_target[mask, idx] * std + float(stats.get("mean", 0.0))
        abs_error = torch.abs(pred_raw - target_raw)
        buffers["reg_abs_error_sum"][idx] += float(abs_error.sum().item())
        buffers["reg_counts"][idx] += int(mask.sum().item())


def summarize_metrics(buffers: dict[str, Any], prefix: str) -> dict[str, float]:
    samples = max(int(buffers["samples"]), 1)
    metrics: dict[str, float] = {f"{prefix}_loss": buffers["loss_sum"] / samples}

    task_f1_values: list[float] = []
    task_acc_values: list[float] = []
    for task in CLASSIFICATION_TASKS:
        y_true = buffers["cls_true"][task.name]
        y_pred = buffers["cls_pred"][task.name]
        if not y_true:
            continue
        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        macro_f1 = macro_f1_score(y_true, y_pred, task.num_classes)
        metrics[f"{prefix}_{task.name}_acc"] = float(acc)
        metrics[f"{prefix}_{task.name}_macro_f1"] = float(macro_f1)
        task_acc_values.append(float(acc))
        task_f1_values.append(float(macro_f1))

    metrics[f"{prefix}_mean_acc"] = float(np.mean(task_acc_values)) if task_acc_values else 0.0
    metrics[f"{prefix}_mean_macro_f1"] = float(np.mean(task_f1_values)) if task_f1_values else 0.0

    mae_values: list[float] = []
    for idx, target in enumerate(REGRESSION_TARGETS):
        count = int(buffers["reg_counts"][idx])
        if count <= 0:
            continue
        mae = float(buffers["reg_abs_error_sum"][idx] / count)
        metrics[f"{prefix}_{target}_mae"] = mae
        mae_values.append(mae)
    metrics[f"{prefix}_mean_reg_mae"] = float(np.mean(mae_values)) if mae_values else 0.0
    return metrics


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: dict[str, torch.Tensor],
    regression_stats: dict[str, dict[str, float]],
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    buffers = empty_metric_buffers()

    for batch in loader:
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        if training:
            optimizer.zero_grad(set_to_none=True)

        use_amp = bool(args.amp and device.type == "cuda")
        with torch.set_grad_enabled(training), autocast_context(use_amp):
            outputs = model(batch["image"])
            cls_loss, _ = classification_loss(
                outputs=outputs,
                targets=batch["cls_targets"],
                class_weights=class_weights,
                label_smoothing=args.label_smoothing,
                ordinal_loss_weight=args.ordinal_loss_weight,
                focal_gamma=args.focal_gamma,
            )
            reg_loss = regression_loss(
                outputs=outputs,
                targets=batch["reg_targets"],
                mask=batch["reg_mask"],
            )
            loss = (args.cls_loss_weight * cls_loss) + (args.reg_loss_weight * reg_loss)
        if training:
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                if args.grad_clip_norm and args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip_norm and args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()

        update_metric_buffers(buffers, outputs, batch, loss, regression_stats)

    prefix = "train" if training else "val"
    return summarize_metrics(buffers, prefix)


def resolve_checkpoint_dir(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir is not None:
        return args.checkpoint_dir
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_CHECKPOINT_ROOT / run_name


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_score: float,
    best_epoch: int,
    args: argparse.Namespace,
    regression_stats: dict[str, dict[str, float]],
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {key: _path_to_str(value) for key, value in vars(args).items()}
    payload = {
        "epoch": epoch,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "grade_scheme": args.grade_scheme,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "config": config,
        "classification_tasks": [
            {
                "name": task.name,
                "column": task.column,
                "part_name": task.part_name,
                "num_classes": task.num_classes,
            }
            for task in CLASSIFICATION_TASKS
        ],
        "regression_targets": list(REGRESSION_TARGETS),
        "regression_stats": regression_stats,
    }
    torch.save(payload, path)


def append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fieldnames = list(row.keys())
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_history_plot(history_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(history_path)
    output_path = history_path.with_suffix(".png")
    if df.empty:
        return output_path

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(df["epoch"], df["train_loss"], label="train_loss")
    axes[0].plot(df["epoch"], df["val_loss"], label="val_loss")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_mean_macro_f1"], label="train_mean_macro_f1")
    axes[1].plot(df["epoch"], df["val_mean_macro_f1"], label="val_mean_macro_f1")
    axes[1].plot(df["epoch"], df["best_score"], linestyle="--", label="best_score")
    axes[1].set_ylabel("Macro F1")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(df["epoch"], df["val_mean_reg_mae"], label="val_mean_reg_mae")
    axes[2].plot(df["epoch"], df["lr"], label="lr")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def maybe_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    resume_path: Path | None,
    device: torch.device,
) -> tuple[int, float, int]:
    if resume_path is None:
        return 0, float("-inf"), -1
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_score = float(checkpoint.get("best_score", float("-inf")))
    best_epoch = int(checkpoint.get("best_epoch", -1))
    logger.info("resumed from %s at epoch=%d", resume_path, start_epoch + 1)
    return start_epoch, best_score, best_epoch


def train(args: argparse.Namespace) -> None:
    global CLASSIFICATION_TASKS
    CLASSIFICATION_TASKS = get_classification_tasks(args.grade_scheme)

    if not args.train_csv.exists():
        raise FileNotFoundError(f"Missing train CSV: {args.train_csv}")
    if not args.val_csv.exists():
        raise FileNotFoundError(f"Missing val CSV: {args.val_csv}")

    split_summary = check_train_val_split(args.train_csv, args.val_csv)
    logger.info("subject-disjoint train/val split: %s", split_summary)

    set_seed(args.seed)
    checkpoint_dir = resolve_checkpoint_dir(args)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    if args.max_train_rows > 0:
        args.train_csv = subset_csv(
            args.train_csv,
            checkpoint_dir / f"train_subset_{args.max_train_rows}.csv",
            args.max_train_rows,
        )
    if args.max_val_rows > 0:
        args.val_csv = subset_csv(
            args.val_csv,
            checkpoint_dir / f"val_subset_{args.max_val_rows}.csv",
            args.max_val_rows,
        )

    regression_stats = build_regression_stats(args.train_csv)
    save_regression_stats(regression_stats, checkpoint_dir / "regression_stats.json")
    train_loader, val_loader = build_dataloaders(args, regression_stats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)
    model = get_model(
        weights=args.weights,
        pretrained_path=args.pretrained_path,
        projection_dim=args.projection_dim,
        dropout_p=args.dropout_p,
        freeze_backbone=args.freeze_backbone,
        freeze_until=args.freeze_until,
        tasks=CLASSIFICATION_TASKS,
    ).to(device)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    logger.info(
        "trainable params=%d / %d (%.2f%%), freeze_backbone=%s, freeze_until=%s",
        trainable_count,
        total_params,
        100.0 * trainable_count / max(total_params, 1),
        args.freeze_backbone,
        args.freeze_until,
    )

    optimizer = build_optimizer(
        model=model,
        learning_rate=args.learning_rate,
        backbone_lr_mult=args.backbone_lr_mult,
        weight_decay=args.weight_decay,
    )
    logger.info(
        "optimizer groups=%s",
        [
            {"lr": group["lr"], "weight_decay": group["weight_decay"], "params": len(group["params"])}
            for group in optimizer.param_groups
        ],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )
    scaler = make_grad_scaler(args.amp and device.type == "cuda")
    class_weights = compute_class_weights(
        args.train_csv,
        device,
        class_weight_power=args.class_weight_power,
        grade_scheme=args.grade_scheme,
    )

    latest_path = checkpoint_dir / "latest.pth"
    best_path = checkpoint_dir / "best.pth"
    history_path = checkpoint_dir / "history.csv"
    start_epoch, best_score, best_epoch = maybe_resume(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        resume_path=args.resume,
        device=device,
    )
    bad_epochs = 0

    logger.info("checkpoint_dir=%s", checkpoint_dir)
    logger.info("grade_scheme=%s", args.grade_scheme)
    logger.info("classification tasks=%s", [task.name for task in CLASSIFICATION_TASKS])
    logger.info("regression targets=%s", list(REGRESSION_TARGETS))

    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            class_weights=class_weights,
            regression_stats=regression_stats,
            args=args,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_metrics = run_epoch(
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
        append_history(history_path, row)
        plot_path = save_history_plot(history_path)

        save_checkpoint(
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
        )
        if improved:
            save_checkpoint(
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
        "best_score": best_score,
        "best_epoch": best_epoch + 1 if best_epoch >= 0 else 0,
    }
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
