"""Repeat DINOv3 skin runs over seeds (and pooling / fine-tuning variants) and report mean ± std.

A 0.02 Macro-F1 gap from a single seed is within run-to-run noise for ~1,000 validation
images, so each configuration is trained with several seeds. Frozen-feature runs share one
embedding cache per pooling, so extra seeds only retrain the small head.

  python notebooks/dy/run_dinov3_seeds.py --seeds 20260510 20260511 20260512 \
      --configs frozen_pooler frozen_cls_mean finetune_last2 --test-csv data/processed/dy_fg_test_metadata.csv \
      --val-csv data/processed/dy_fg_val2_metadata.csv
  python notebooks/dy/run_dinov3_seeds.py ... --dry-run      # print commands
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "trained" / "dy_forehead_glabella"

CONFIGS = {
    "frozen_pooler": ("train_dinov3.py", ["--pooling", "pooler"]),          # = the 0.7067 run (v1 settings)
    "frozen_mean": ("train_dinov3.py", ["--pooling", "mean"]),
    "frozen_cls_mean": ("train_dinov3.py", ["--pooling", "cls_mean"]),
    "frozen_aug_cls_mean": ("train_dinov3_finetune.py", ["--unfreeze-last-n", "0", "--pooling", "cls_mean",
                                                         "--weighted-sampler"]),
    "finetune_last2": ("train_dinov3_finetune.py", ["--unfreeze-last-n", "2", "--pooling", "cls_mean",
                                                    "--weighted-sampler", "--amp"]),
    "finetune_last4": ("train_dinov3_finetune.py", ["--unfreeze-last-n", "4", "--pooling", "cls_mean",
                                                    "--weighted-sampler", "--amp"]),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260510, 20260511, 20260512])
    parser.add_argument("--configs", nargs="+", choices=sorted(CONFIGS), default=["frozen_pooler", "frozen_cls_mean"])
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--val-csv", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--prefix", default="seeds")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, default=CHECKPOINT_ROOT / "seed_summary")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cache_dir = CHECKPOINT_ROOT / "_embedding_cache"
    results = {}
    for config in args.configs:
        script, extra = CONFIGS[config]
        for seed in args.seeds:
            run = f"{args.prefix}_{config}_s{seed}"
            command = [args.python, str(HERE / script), "--run-name", run, "--seed", str(seed),
                       "--grade-scheme", "three", *extra]
            for flag, value in (("--train-csv", args.train_csv), ("--val-csv", args.val_csv),
                                ("--test-csv", args.test_csv), ("--model-revision", args.model_revision)):
                if value:
                    command += [flag, str(value)]
            if script == "train_dinov3.py":
                command += ["--embedding-cache-dir", str(cache_dir)]
            summary_path = CHECKPOINT_ROOT / run / "training_summary.json"
            print(" ".join(command))
            if args.dry_run:
                continue
            if not summary_path.is_file():
                subprocess.run(command, check=True, cwd=PROJECT_ROOT)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            results.setdefault(config, []).append({"seed": seed, "val": summary["best_score"],
                                                   "test": summary.get("test_mean_macro_f1")})
    if args.dry_run or not results:
        return
    lines = ["| config | seeds | val Macro-F1 mean ± sd | test Macro-F1 mean ± sd |", "|---|---:|---:|---:|"]

    def band(values):
        values = [v for v in values if v is not None]
        if not values:
            return "–"
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        return f"{statistics.mean(values):.4f} ± {sd:.4f}"
    for config, runs in results.items():
        lines.append(f"| {config} | {len(runs)} | {band([r['val'] for r in runs])} | {band([r['test'] for r in runs])} |")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "seed_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.output / "seed_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
