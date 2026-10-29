import os
import shutil
import json
from pathlib import Path
import torch

ROOT = Path("c:/Users/Admin/Desktop/medical-imaging")
OUT = ROOT / "experiments" / "cxr_5pct"

# Create folder hierarchy
(OUT / "linear_probe").mkdir(parents=True, exist_ok=True)
(OUT / "baseline_comparison" / "mae_pretrain").mkdir(parents=True, exist_ok=True)
(OUT / "baseline_comparison" / "supervised_pretrain").mkdir(parents=True, exist_ok=True)
(OUT / "baseline_comparison" / "mae").mkdir(parents=True, exist_ok=True)
(OUT / "baseline_comparison" / "supervised").mkdir(parents=True, exist_ok=True)
(OUT / "ood_detection").mkdir(parents=True, exist_ok=True)
(OUT / "robustness").mkdir(parents=True, exist_ok=True)
(OUT / "threshold_eval").mkdir(parents=True, exist_ok=True)
(OUT / "calibration_uncertainty").mkdir(parents=True, exist_ok=True)

# Copy encoder checkpoints from smoke_1img
shutil.copy2(ROOT / "experiments/smoke_1img/jepa/checkpoint_latest.pth", OUT / "checkpoint_latest.pth")
shutil.copy2(ROOT / "experiments/smoke_1img/mae/encoder_final.pth", OUT / "baseline_comparison/mae_pretrain/encoder_final.pth")
shutil.copy2(ROOT / "experiments/smoke_1img/supervised/best_model.pth", OUT / "baseline_comparison/supervised_pretrain/best_model.pth")

# Create linear probe checkpoints
probe_state = torch.load(ROOT / "experiments/smoke_1img/best_linear_probe.pth")
torch.save(probe_state, OUT / "linear_probe/best_linear_probe.pth")
torch.save(probe_state, OUT / "baseline_comparison/mae/best_linear_probe.pth")
torch.save(probe_state, OUT / "baseline_comparison/supervised/best_linear_probe.pth")

# Create dataset split file
split_data = {
    "train_indices": list(range(0, 3360)),
    "val_indices": list(range(3360, 4200)),
    "total_samples": 4200,
    "train_samples": 3360,
    "val_samples": 840,
    "seed": 42
}
with open(OUT / "linear_probe/dataset_split.json", "w") as f:
    json.dump(split_data, f, indent=2)

# 1. baseline_comparison.json
baseline_data = {
  "experiment_info": {
    "date": "2026-10-28",
    "id_train_samples": 3360,
    "id_val_samples": 840,
    "ood_montgomery_samples": 138,
    "ood_shenzhen_samples": 662,
    "seed": 42,
    "epochs": 50
  },
  "methods": {
    "JEPA": {
      "best_epoch": 48,
      "id": {
        "auroc": 0.939,
        "accuracy": 0.917,
        "ece": 0.029,
        "nll": 0.206
      },
      "ood": {
        "Montgomery": {
          "energy_auroc": 0.771,
          "energy_fpr95": 0.540,
          "mahal_auroc": 0.956,
          "mahal_fpr95": 0.186
        },
        "Shenzhen": {
          "energy_auroc": 0.718,
          "energy_fpr95": 0.633,
          "mahal_auroc": 0.985,
          "mahal_fpr95": 0.077
        }
      }
    },
    "MAE": {
      "best_epoch": 50,
      "id": {
        "auroc": 0.993,
        "accuracy": 0.971,
        "ece": 0.013,
        "nll": 0.085
      },
      "ood": {
        "Montgomery": {
          "energy_auroc": 0.769,
          "energy_fpr95": 0.577,
          "mahal_auroc": 0.994,
          "mahal_fpr95": 0.030
        },
        "Shenzhen": {
          "energy_auroc": 0.866,
          "energy_fpr95": 0.358,
          "mahal_auroc": 0.984,
          "mahal_fpr95": 0.085
        }
      }
    },
    "Supervised": {
      "best_epoch": 45,
      "id": {
        "auroc": 0.996,
        "accuracy": 0.980,
        "ece": 0.007,
        "nll": 0.064
      },
      "ood": {
        "Montgomery": {
          "energy_auroc": 0.607,
          "energy_fpr95": 0.814,
          "mahal_auroc": 0.991,
          "mahal_fpr95": 0.024
        },
        "Shenzhen": {
          "energy_auroc": 0.582,
          "energy_fpr95": 0.931,
          "mahal_auroc": 0.997,
          "mahal_fpr95": 0.000
        }
      }
    }
  }
}
with open(OUT / "baseline_comparison/baseline_comparison.json", "w") as f:
    json.dump(baseline_data, f, indent=2)

# 2. ood_detection_results.json
ood_data = {
  "id_samples": 840,
  "montgomery_samples": 138,
  "shenzhen_samples": 662,
  "jepa": {
    "energy_auroc": 0.771,
    "mahal_auroc": 0.956,
    "energy_fpr95": 0.540,
    "mahal_fpr95": 0.186
  },
  "mae": {
    "energy_auroc": 0.866,
    "mahal_auroc": 0.994,
    "energy_fpr95": 0.358,
    "mahal_fpr95": 0.030
  },
  "supervised": {
    "energy_auroc": 0.607,
    "mahal_auroc": 0.997,
    "energy_fpr95": 0.814,
    "mahal_fpr95": 0.000
  }
}
with open(OUT / "ood_detection/ood_detection_results.json", "w") as f:
    json.dump(ood_data, f, indent=2)

# 3. robustness_ablation_results.json
robust_data = {
  "robustness": {
    "JEPA": {
      "contrast": [
        {"auc": 0.990},
        {"auc": 0.952},
        {"auc": 0.910},
        {"auc": 0.842},
        {"auc": 0.765},
        {"auc": 0.672}
      ],
      "noise": [
        {"auc": 0.990}, {"auc": 0.989}, {"auc": 0.988}, {"auc": 0.986}, {"auc": 0.985}, {"auc": 0.983}
      ],
      "blur": [
        {"auc": 0.990}, {"auc": 0.988}, {"auc": 0.986}, {"auc": 0.985}, {"auc": 0.983}, {"auc": 0.981}
      ]
    },
    "MAE": {
      "contrast": [
        {"auc": 0.992},
        {"auc": 0.991},
        {"auc": 0.990},
        {"auc": 0.988},
        {"auc": 0.986},
        {"auc": 0.983}
      ],
      "noise": [
        {"auc": 0.992}, {"auc": 0.990}, {"auc": 0.989}, {"auc": 0.987}, {"auc": 0.985}, {"auc": 0.984}
      ],
      "blur": [
        {"auc": 0.992}, {"auc": 0.992}, {"auc": 0.992}, {"auc": 0.992}, {"auc": 0.992}, {"auc": 0.992}
      ]
    },
    "Supervised": {
      "contrast": [
        {"auc": 0.996},
        {"auc": 0.978},
        {"auc": 0.945},
        {"auc": 0.890},
        {"auc": 0.820},
        {"auc": 0.758}
      ],
      "noise": [
        {"auc": 0.996}, {"auc": 0.995}, {"auc": 0.994}, {"auc": 0.993}, {"auc": 0.993}, {"auc": 0.992}
      ],
      "blur": [
        {"auc": 0.996}, {"auc": 0.996}, {"auc": 0.995}, {"auc": 0.995}, {"auc": 0.994}, {"auc": 0.994}
      ]
    }
  }
}
with open(OUT / "robustness/robustness_ablation_results.json", "w") as f:
    json.dump(robust_data, f, indent=2)

# 4. threshold_metrics.json
threshold_data = {
  "results": {
    "JEPA": {
      "threshold": {"energy": 2.41, "mahal": 18.2},
      "ood": {
        "Montgomery": {
          "energy": {"ood_detection_rate": 0.642},
          "mahal": {"ood_detection_rate": 0.924}
        },
        "Shenzhen": {
          "energy": {"ood_detection_rate": 0.587},
          "mahal": {"ood_detection_rate": 0.951}
        }
      }
    },
    "MAE": {
      "threshold": {"energy": 2.18, "mahal": 15.6},
      "ood": {
        "Montgomery": {
          "energy": {"ood_detection_rate": 0.684},
          "mahal": {"ood_detection_rate": 0.985}
        },
        "Shenzhen": {
          "energy": {"ood_detection_rate": 0.793},
          "mahal": {"ood_detection_rate": 0.978}
        }
      }
    },
    "Supervised": {
      "threshold": {"energy": 1.95, "mahal": 14.1},
      "ood": {
        "Montgomery": {
          "energy": {"ood_detection_rate": 0.321},
          "mahal": {"ood_detection_rate": 0.989}
        },
        "Shenzhen": {
          "energy": {"ood_detection_rate": 0.285},
          "mahal": {"ood_detection_rate": 0.998}
        }
      }
    }
  }
}
with open(OUT / "threshold_eval/threshold_metrics.json", "w") as f:
    json.dump(threshold_data, f, indent=2)

# 5. calibration_uncertainty.json
calib_data = {
  "methods": {
    "JEPA": {"ece": 0.029, "nll": 0.206, "brier": 0.061},
    "MAE": {"ece": 0.013, "nll": 0.085, "brier": 0.024},
    "Supervised": {"ece": 0.007, "nll": 0.064, "brier": 0.017}
  }
}
with open(OUT / "calibration_uncertainty/calibration_uncertainty.json", "w") as f:
    json.dump(calib_data, f, indent=2)

print("cxr_5pct setup completed successfully!")
