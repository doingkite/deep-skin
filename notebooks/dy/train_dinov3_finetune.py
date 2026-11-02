"""Partially fine-tune DINOv3 (last N transformer blocks) for forehead/glabella grading.

Fairer counterpart to the ResNet-50 run: the ResNet used augmentation, a weighted
sampler, ordinal loss and a partly trainable backbone, while train_dinov3.py keeps
DINOv3 frozen and trains a head on cached features without augmentation. This script
gives DINOv3 the same advantages, so the comparison isolates the backbone.

  python notebooks/dy/train_dinov3_finetune.py --run-name dy_dinov3_ft_last2 --unfreeze-last-n 2 \
      --weighted-sampler --pooling cls_mean --seed 20260511 --test-csv data/processed/..._test_metadata.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

import train as train_utils
from data_checks import check_train_val_split
from dataset import ForeheadGlabellaDataset, get_transforms
from task_config import REGRESSION_TARGETS, get_classification_tasks
from train_dinov3 import (DEFAULT_CHECKPOINT_ROOT, DEFAULT_TRAIN_CSV, DEFAULT_VAL_CSV,
                          DinoEmbeddingMultiTask, encoder_forward, pooled_features, resolve_token)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_CSV)
    p.add_argument("--test-csv", type=Path, default=None)
    p.add_argument("--run-name", default="dy_dinov3_vits16_finetune_v1")
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--model-name", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    p.add_argument("--model-revision", default=None)
    p.add_argument("--hf-token", default="")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--grade-scheme", choices=["original", "three"], default="three")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--pooling", choices=["pooler", "cls", "mean", "cls_mean"], default="cls_mean")
    p.add_argument("--unfreeze-last-n", type=int, default=2, help="0 = frozen backbone (head only, with augmentation)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=5e-4, help="head learning rate")
    p.add_argument("--backbone-lr-mult", type=float, default=0.05, help="backbone lr = lr * mult")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--projection-dim", type=int, default=512)
    p.add_argument("--dropout-p", type=float, default=0.35)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--focal-gamma", type=float, default=1.0)
    p.add_argument("--ordinal-loss-weight", type=float, default=0.2)
    p.add_argument("--cls-loss-weight", type=float, default=1.0)
    p.add_argument("--reg-loss-weight", type=float, default=0.15)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--weighted-sampler", action="store_true")
    p.add_argument("--sampler-power", type=float, default=0.5)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--early-stopping-patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260511)
    p.add_argument("--max-train-rows", type=int, default=0)
    p.add_argument("--max-val-rows", type=int, default=0)
    return p.parse_args(argv)


def transformer_blocks(backbone: nn.Module) -> nn.ModuleList:
    for path in ("layer", "encoder.layer", "blocks", "model.layer"):
        module = backbone
        for part in path.split("."):
            module = getattr(module, part, None)
            if module is None:
                break
        if isinstance(module, nn.ModuleList) and len(module):
            return module
    raise AttributeError("Could not find the transformer block list of the DINOv3 model")


class DinoFinetune(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, pooling: str, unfreeze_last_n: int) -> None:
        super().__init__()
        self.backbone, self.head, self.pooling = backbone, head, pooling
        self.registers = int(getattr(backbone.config, "num_register_tokens", 0) or 0)
        for param in self.backbone.parameters():
            param.requires_grad = False
        blocks = transformer_blocks(self.backbone)
        if unfreeze_last_n > len(blocks):
            raise ValueError(f"--unfreeze-last-n {unfreeze_last_n} > {len(blocks)} blocks")
        if unfreeze_last_n > 0:
            for block in blocks[len(blocks) - unfreeze_last_n:]:
                for param in block.parameters():
                    param.requires_grad = True
            for name in ("norm", "layernorm"):
                norm = getattr(self.backbone, name, None)
                if isinstance(norm, nn.Module):
                    for param in norm.parameters():
                        param.requires_grad = True
        self.trainable_backbone = unfreeze_last_n > 0

    def train(self, mode: bool = True) -> "DinoFinetune":
        super().train(mode)
        if not self.trainable_backbone:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> dict:
        with torch.set_grad_enabled(self.trainable_backbone and torch.is_grad_enabled()):
            outputs = encoder_forward(self.backbone, images)
        return self.head(pooled_features(outputs, self.pooling, self.registers))


def run_epoch(model, loader, device, class_weights, regression_stats, args, optimizer=None, scaler=None, prefix=None):
    training = optimizer is not None
    model.train(training)
    buffers = train_utils.empty_metric_buffers()
    use_amp = bool(args.amp and device.type == "cuda")
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), train_utils.autocast_context(use_amp):
            outputs = model(batch["image"])
            cls_loss, _ = train_utils.classification_loss(
                outputs=outputs, targets=batch["cls_targets"], class_weights=class_weights,
                label_smoothing=args.label_smoothing, ordinal_loss_weight=args.ordinal_loss_weight,
                focal_gamma=args.focal_gamma)
            reg_loss = train_utils.regression_loss(outputs=outputs, targets=batch["reg_targets"], mask=batch["reg_mask"])
            loss = args.cls_loss_weight * cls_loss + args.reg_loss_weight * reg_loss
        if training:
            scaler.scale(loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        outputs = {"cls": {k: v.float() for k, v in outputs["cls"].items()}, "reg": outputs["reg"].float()}
        train_utils.update_metric_buffers(buffers, outputs, batch, loss.float(), regression_stats)
    return train_utils.summarize_metrics(buffers, prefix or ("train" if training else "val"))


def make_loader(csv_path, args, regression_stats, tasks, split, shuffle, sampler=None):
    transform = get_transforms(args.image_size, "train" if (split == "train" and not args.no_augment) else "val")
    dataset = ForeheadGlabellaDataset(csv_path, transform=transform, regression_stats=regression_stats,
                                      tasks=tasks, grade_scheme=args.grade_scheme)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=(split == "train"))  # BatchNorm in the head cannot take a batch of 1


def train(args: argparse.Namespace) -> dict:
    from transformers import AutoModel

    tasks = get_classification_tasks(args.grade_scheme)
    train_utils.CLASSIFICATION_TASKS = tasks
    logger.info("train/val split: %s", check_train_val_split(args.train_csv, args.val_csv))
    if args.test_csv is not None:
        logger.info("train/test split: %s", check_train_val_split(args.train_csv, args.test_csv))
        logger.info("val/test split: %s", check_train_val_split(args.val_csv, args.test_csv))
    train_utils.set_seed(args.seed)
    checkpoint_dir = args.checkpoint_dir or DEFAULT_CHECKPOINT_ROOT / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.max_train_rows > 0:
        args.train_csv = train_utils.subset_csv(args.train_csv, checkpoint_dir / "train_subset.csv", args.max_train_rows)
    if args.max_val_rows > 0:
        args.val_csv = train_utils.subset_csv(args.val_csv, checkpoint_dir / "val_subset.csv", args.max_val_rows)
    regression_stats = train_utils.build_regression_stats(args.train_csv)
    train_utils.save_regression_stats(regression_stats, checkpoint_dir / "regression_stats.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = AutoModel.from_pretrained(args.model_name, revision=args.model_revision,
                                         token=resolve_token(args.hf_token), local_files_only=args.local_files_only)
    hidden = int(backbone.config.hidden_size)
    feature_dim = 2 * hidden if args.pooling == "cls_mean" else hidden
    head = DinoEmbeddingMultiTask(feature_dim, args.projection_dim, args.dropout_p, tasks)
    model = DinoFinetune(backbone, head, args.pooling, args.unfreeze_last_n).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("device=%s trainable params=%d (backbone blocks unfrozen: %d)", device, trainable, args.unfreeze_last_n)

    sampler = None
    if args.weighted_sampler:
        weights = train_utils.build_sample_weights(args.train_csv, sampler_power=args.sampler_power,
                                                   grade_scheme=args.grade_scheme)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = make_loader(args.train_csv, args, regression_stats, tasks, "train", True, sampler)
    val_loader = make_loader(args.val_csv, args, regression_stats, tasks, "val", False)
    optimizer = train_utils.build_optimizer(model, args.learning_rate, args.backbone_lr_mult, args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-7)
    scaler = train_utils.make_grad_scaler(bool(args.amp and device.type == "cuda"))
    class_weights = train_utils.compute_class_weights(args.train_csv, device, grade_scheme=args.grade_scheme,
                                                      class_weight_power=args.class_weight_power)
    best_score, best_epoch, bad = float("-inf"), -1, 0
    best_path = checkpoint_dir / "best.pth"
    history = checkpoint_dir / "history.csv"
    for epoch in range(args.epochs):
        train_m = run_epoch(model, train_loader, device, class_weights, regression_stats, args, optimizer, scaler)
        val_m = run_epoch(model, val_loader, device, class_weights, regression_stats, args)
        score = val_m["val_mean_macro_f1"]
        scheduler.step(score)
        if score > best_score + 1e-4:
            best_score, best_epoch, bad = score, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_score": score,
                        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                                   if k != "hf_token"},
                        "regression_stats": regression_stats, "regression_targets": list(REGRESSION_TARGETS)},
                       best_path)
        else:
            bad += 1
        train_utils.append_history(history, {"epoch": epoch + 1, **train_m, **val_m, "best_score": best_score})
        logger.info("epoch %d train_loss=%.4f val_macro_f1=%.4f best=%.4f", epoch + 1,
                    train_m["train_loss"], score, best_score)
        if args.early_stopping_patience > 0 and bad >= args.early_stopping_patience:
            break
    summary = {"checkpoint_dir": str(checkpoint_dir), "best_score": best_score, "best_epoch": best_epoch + 1,
               "pooling": args.pooling, "unfreeze_last_n": args.unfreeze_last_n, "seed": args.seed,
               "augment": not args.no_augment, "weighted_sampler": args.weighted_sampler}
    if args.test_csv is not None:
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model_state_dict"])
        test_loader = make_loader(args.test_csv, args, regression_stats, tasks, "test", False)
        test_m = run_epoch(model, test_loader, device, class_weights, regression_stats, args, prefix="test")
        summary.update({"test_csv": str(args.test_csv), "test_metrics": test_m,
                        "test_mean_macro_f1": test_m["test_mean_macro_f1"]})
    (checkpoint_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                          encoding="utf-8")
    logger.info("summary=%s", json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    train(parse_args())


if __name__ == "__main__":
    main()
