"""ResNet-50 multitask model for forehead and glabella analysis."""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from task_config import CLASSIFICATION_TASKS, REGRESSION_TARGETS, ClassificationTask


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_RESNET50 = PROJECT_ROOT / "checkpoints" / "pretrained" / "resnet50-0676ba61.pth"


def _clean_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a state_dict or contain state_dict")

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if not isinstance(value, torch.Tensor):
            continue
        name = key
        for prefix in ("module.", "backbone."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        cleaned[name] = value
    return cleaned


def build_resnet50_backbone(
    weights: str = "local",
    pretrained_path: Path | str | None = DEFAULT_LOCAL_RESNET50,
) -> models.ResNet:
    """Create a ResNet-50 backbone and load optional pretrained weights."""
    backbone = models.resnet50(weights=None)

    if weights == "none":
        return backbone

    if weights in {"local", "local_or_none"}:
        path = Path(pretrained_path) if pretrained_path else DEFAULT_LOCAL_RESNET50
        if path.exists():
            payload = torch.load(path, map_location="cpu")
            backbone.load_state_dict(_clean_state_dict(payload), strict=True)
        elif weights == "local":
            raise FileNotFoundError(f"Missing pretrained ResNet-50 weights: {path}")
        else:
            warnings.warn(
                f"No pretrained weights at {path}; starting ResNet-50 from random weights. "
                "Use weights='local' to require pretrained weights.",
                RuntimeWarning,
                stacklevel=2,
            )
        return backbone

    if weights == "torchvision":
        tv_weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=tv_weights)
        return backbone

    raise ValueError("weights must be one of: local_or_none, local, torchvision, none")


class ResNet50MultiTask(nn.Module):
    """Shared ResNet-50 encoder with classification and regression heads."""

    def __init__(
        self,
        weights: str = "local",
        pretrained_path: Path | str | None = DEFAULT_LOCAL_RESNET50,
        projection_dim: int = 512,
        dropout_p: float = 0.30,
        freeze_backbone: bool = False,
        freeze_until: str = "none",
        tasks: tuple[ClassificationTask, ...] = CLASSIFICATION_TASKS,
    ):
        super().__init__()
        self.tasks = tasks
        backbone = build_resnet50_backbone(weights=weights, pretrained_path=pretrained_path)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.shared_head = nn.Sequential(
            nn.Linear(in_features, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )
        self.classification_heads = nn.ModuleDict(
            {
                task.name: nn.Linear(projection_dim, task.num_classes)
                for task in self.tasks
            }
        )
        self.regression_head = nn.Linear(projection_dim, len(REGRESSION_TARGETS))

        if freeze_backbone:
            self.freeze_backbone()
        elif freeze_until != "none":
            self.freeze_backbone_until(freeze_until)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def freeze_backbone_until(self, freeze_until: str) -> None:
        """Freeze early ResNet stages while leaving later stages trainable."""
        stage_order = ["stem", "layer1", "layer2", "layer3"]
        if freeze_until not in stage_order:
            raise ValueError(f"freeze_until must be one of {stage_order} or 'none'")

        selected = set(stage_order[: stage_order.index(freeze_until) + 1])
        for name, param in self.backbone.named_parameters():
            stage = None
            if name.startswith(("conv1.", "bn1.")):
                stage = "stem"
            elif name.startswith("layer1."):
                stage = "layer1"
            elif name.startswith("layer2."):
                stage = "layer2"
            elif name.startswith("layer3."):
                stage = "layer3"

            if stage in selected:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        features = self.backbone(x)
        shared = self.shared_head(features)
        cls_outputs = {
            name: head(shared)
            for name, head in self.classification_heads.items()
        }
        reg_outputs = self.regression_head(shared)
        return {"cls": cls_outputs, "reg": reg_outputs}


def get_model(
    weights: str = "local",
    pretrained_path: Path | str | None = DEFAULT_LOCAL_RESNET50,
    projection_dim: int = 512,
    dropout_p: float = 0.30,
    freeze_backbone: bool = False,
    freeze_until: str = "none",
    tasks: tuple[ClassificationTask, ...] = CLASSIFICATION_TASKS,
) -> ResNet50MultiTask:
    return ResNet50MultiTask(
        weights=weights,
        pretrained_path=pretrained_path,
        projection_dim=projection_dim,
        dropout_p=dropout_p,
        freeze_backbone=freeze_backbone,
        freeze_until=freeze_until,
        tasks=tasks,
    )
