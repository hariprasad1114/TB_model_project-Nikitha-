#!/usr/bin/env python3
"""
Full pipeline runner - runs all 7 stages sequentially and logs progress.
Optimized for CPU-only runs (batch_size=16, num_workers=2).
Usage: python run_full_pipeline.py
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOG_DIR = ROOT / "logs" / "full_run"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = ROOT / "Datasets" / "CXR"
TB_ROOT = DATA_ROOT / "TB_Chest_Radiography_Database"
OUT = ROOT / "experiments" / "cxr_jepa_pilot"
BC_DIR = OUT / "baseline_comparison"

JEPA_CKPT = OUT / "checkpoint_latest.pth"
MAE_ENC = BC_DIR / "mae_pretrain" / "encoder_final.pth"
SUP_BEST = BC_DIR / "supervised_pretrain" / "best_model.pth"
PROBE_DIR = OUT / "linear_probe"
PROBE_CKPT = PROBE_DIR / "best_linear_probe.pth"
SPLIT_FILE = PROBE_DIR / "dataset_split.json"

STAGES = [
    {
        "name": "jepa_pretrain",
        "cmd": [PYTHON, "cxr_ood/pretraining/jepa_pretrain.py",
                "--config", "cxr_ood/configs/cxr_vit_small.yaml",
                "--device", "cpu"],
        "output": JEPA_CKPT,
    },
    {
        "name": "mae_pretrain",
        "cmd": [PYTHON, "cxr_ood/pretraining/mae_pretrain.py",
                "--data-root", str(TB_ROOT),
                "--output-dir", str(BC_DIR / "mae_pretrain"),
                "--batch-size", "16",
                "--seed", "42"],
        "output": MAE_ENC,
    },
    {
        "name": "supervised_pretrain",
        "cmd": [PYTHON, "cxr_ood/pretraining/supervised_pretrain.py",
                "--data-root", str(TB_ROOT),
                "--output-dir", str(BC_DIR / "supervised_pretrain"),
                "--batch-size", "16",
                "--seed", "42"],
        "output": SUP_BEST,
    },
    {
        "name": "linear_probe",
        "cmd": [PYTHON, "cxr_ood/evaluation/linear_probe.py",
                "--checkpoint", str(JEPA_CKPT),
                "--data-path", str(TB_ROOT),
                "--output-dir", str(PROBE_DIR),
                "--batch-size", "16",
                "--seed", "42"],
        "output": PROBE_CKPT,
    },
    {
        "name": "ood_detection",
        "cmd": [PYTHON, "cxr_ood/evaluation/ood_detection.py",
                "--jepa-checkpoint", str(JEPA_CKPT),
                "--probe-checkpoint", str(PROBE_CKPT),
                "--data-root", str(DATA_ROOT),
                "--split-file", str(SPLIT_FILE),
                "--output-dir", str(OUT / "ood_detection"),
                "--batch-size", "16",
                "--seed", "42"],
        "output": OUT / "ood_detection" / "ood_detection_results.json",
    },
    {
        "name": "baseline_comparison",
        "cmd": [PYTHON, "cxr_ood/analysis/baseline_comparison.py",
                "--jepa-checkpoint", str(JEPA_CKPT),
                "--mae-checkpoint", str(MAE_ENC),
                "--supervised-checkpoint", str(SUP_BEST),
                "--data-root", str(DATA_ROOT),
                "--output-dir", str(BC_DIR),
                "--batch-size", "16",
                "--seed", "42"],
        "output": BC_DIR / "baseline_comparison.json",
    },
    {
        "name": "robustness_ablation",
        "cmd": [PYTHON, "cxr_ood/analysis/robustness_ablation.py",
                "--jepa-checkpoint", str(JEPA_CKPT),
                "--mae-checkpoint", str(MAE_ENC),
                "--supervised-checkpoint", str(SUP_BEST),
                "--data-root", str(DATA_ROOT),
                "--output-dir", str(OUT / "robustness"),
                "--batch-size", "16",
                "--seed", "42"],
        "output": OUT / "robustness" / "robustness_ablation_results.json",
    },
]


def run_stage(stage):
    name = stage["name"]
    output = Path(stage["output"])
    log_file = LOG_DIR / f"{name}.log"

    # Skip if output already exists
    if output.exists():
        print(f"\n[SKIP] {name}: output already exists at {output}")
        return "skipped", 0.0

    print(f"\n{'='*60}")
    print(f"[START] {name} @ {datetime.now().strftime('%H:%M:%S')}")
    print(f"  cmd: {' '.join(str(c) for c in stage['cmd'])}")
    print(f"  log: {log_file}")
    print(f"{'='*60}")

    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            [str(c) for c in stage["cmd"]],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
        proc.wait()

    dur = time.time() - t0
    if proc.returncode != 0 or not output.exists():
        print(f"\n[FAIL] {name} failed (exit={proc.returncode}, output_exists={output.exists()})")
        print(f"  See log: {log_file}")
        return "failed", dur

    print(f"\n[DONE] {name} in {dur/60:.1f}min")
    return "done", dur


def main():
    print(f"Full pipeline started @ {datetime.now().isoformat()}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Output dir: {OUT}")
    summary = []

    for stage in STAGES:
        status, dur = run_stage(stage)
        summary.append({"stage": stage["name"], "status": status, "duration_min": dur / 60})
        if status == "failed":
            print(f"\n[ABORT] Pipeline stopped at {stage['name']}. Fix and re-run.")
            break

    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'stage':<25} {'status':<10} {'min':>8}")
    print(f"{'-'*45}")
    for s in summary:
        print(f"{s['stage']:<25} {s['status']:<10} {s['duration_min']:>8.1f}")
    print(f"{'='*60}")

    result_file = OUT / "pipeline_summary.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "summary": summary}, f, indent=2)
    print(f"\nSummary saved to {result_file}")


if __name__ == "__main__":
    main()
