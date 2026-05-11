# 🧪 TFM-Bench: Benchmarking Tabular Foundation Models on Credit Data

> A reproducible, open-source benchmarking suite that evaluates **every major Tabular Foundation Model** on credit risk datasets — from zero-shot inference to fine-tuning — with detailed analysis of accuracy, speed, scalability, and practical limitations.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Quick Start

### Installation

```bash
# Minimal install — no GPU required, works with sklearn baselines + XGBoost/LightGBM
pip install -e ".[gbdt]"

# All optional model groups
pip install -e ".[gbdt,tabpfn,tabicl,deep]"

# Only CPU / no-GPU dependencies (sklearn baselines only)
pip install -e "."
```

> **Note:** TabPFN v2.5 requires a HuggingFace account and license acceptance.
> Run `huggingface-cli login` and visit the model page to accept terms.

### 5-Line Example

```python
from tfm_benchmark import load_dataset, run_benchmark

X_train, X_test, y_train, y_test = load_dataset("german_credit")
results = run_benchmark(X_train, y_train, X_test, y_test,
                        models=["random_forest", "logistic_regression"])
print(results[["model_name", "auc_roc", "fit_time"]])
```

### Using Your Own Data

```python
from tfm_benchmark import load_dataset, run_benchmark, list_models

# Load from a CSV (auto-splits 80/20, stratified)
X_train, X_test, y_train, y_test = load_dataset("my_data.csv", target="default")

# Or pass a DataFrame directly
import pandas as pd
df = pd.read_csv("my_data.csv")
X_train, X_test, y_train, y_test = load_dataset(df, target="default")

# See all available models (including optional ones)
print(list_models())

# Run benchmark
results = run_benchmark(X_train, y_train, X_test, y_test, models="auto")
results.to_csv("my_results.csv", index=False)
```

### Class-Based API

```python
from tfm_benchmark import Benchmarker

b = Benchmarker(models=["random_forest", "logistic_regression"],
                dataset_name="german_credit")
results = b.fit_evaluate(X_train, y_train, X_test, y_test)
b.plot_leaderboard(save_path="leaderboard.png", show=False)
b.save_results("results/run.csv")
```

### Runnable Examples

```bash
# Benchmark on bundled German Credit data — saves PNG + CSV
python examples/01_quick_start.py

# Benchmark on a generated synthetic CSV (or pass --csv your_data.csv)
python examples/02_custom_data.py --models random_forest logistic_regression
```

---

## 🎯 What This Repo Does

1. **Loads** real-world credit datasets (Give Me Some Credit, German Credit, Taiwan Credit)
2. **Benchmarks 10+ models** on raw data with zero preprocessing
3. **Fine-tunes** models that support it and re-evaluates
4. **Scales** the same dataset from 1K → 10K → 50K → 150K rows to test limits
5. **Produces** publication-ready comparison tables, charts, and a leaderboard
6. **Documents** every limitation, gotcha, and license restriction

---

## 📊 Models Benchmarked

| Model | Version | Type | Max Rows | Max Features | License | Fine-Tunable | GPU Required |
|-------|---------|------|----------|--------------|---------|-------------|--------------|
| **TabPFN v1** | 1.0 | ICL Foundation | 1,000 | 100 | Apache 2.0 | ❌ | Optional |
| **TabPFN v2** | 2.0 | ICL Foundation | 10,000 | 500 | Prior Labs (commercial OK w/ attribution) | ✅ | Recommended |
| **TabPFN v2.5** | 2.5 | ICL Foundation | 50,000 | 2,000 | Non-commercial (research/eval only) | ✅ | Yes |
| **Real-TabPFN-2.5** | 2.5-real | ICL Foundation (real-data finetuned) | 50,000 | 2,000 | Non-commercial | ✅ | Yes |
| **TabICL v1.1** | 1.1 | ICL Foundation | ~50,000 | ~500 | BSD-3-Clause | ❌ | Recommended |
| **TabICLv2** | 2.0 | ICL Foundation | 1,000,000 | 2,000 | BSD-3-Clause | ❌ | Recommended |
| **Mitra** | 1.0 | ICL Foundation | ~10,000 | ~100 | Apache 2.0 | ❌ | Yes |
| **TabDPT** | 1.0 | ICL + SSL | ~10,000 | ~500 | MIT | ❌ | Yes |
| **TabNet** | - | Attention-based DL | Unlimited | Unlimited | Apache 2.0 / MIT | ✅ (from scratch) | Optional |
| **FT-Transformer** | - | Feature Tokenizer + Transformer | Unlimited | Unlimited | MIT | ✅ (from scratch) | Yes |
| **XGBoost** | - | GBDT (Baseline) | Unlimited | Unlimited | Apache 2.0 | N/A | ❌ |
| **CatBoost** | - | GBDT (Baseline) | Unlimited | Unlimited | Apache 2.0 | N/A | Optional |
| **LightGBM** | - | GBDT (Baseline) | Unlimited | Unlimited | MIT | N/A | ❌ |
| **Random Forest** | - | Ensemble (Baseline) | Unlimited | Unlimited | BSD-3 (sklearn) | N/A | ❌ |

### ⚠️ Critical License Notes

```
TabPFN v2.5 / Real-TabPFN-2.5:
  → Non-commercial license. You CANNOT use for production, revenue, or business decisions.
  → Fine-tuned derivatives inherit the same restriction.
  → Contact sales@priorlabs.ai for commercial license.

TabPFN v2:
  → Commercial use allowed WITH attribution ("Built with PriorLabs-TabPFN").

TabICLv2:
  → BSD-3-Clause — fully open, fully commercial. Best enterprise option.

Mitra:
  → Apache 2.0 — fully open, fully commercial. Native AutoGluon integration.
```

---

## 📁 Project Structure

```
tfm-benchmark/
├── README.md                          # You are here
├── LICENSE                            # MIT
├── pyproject.toml                     # Dependencies & project config
├── requirements.txt                   # Pinned dependencies
│
├── configs/
│   ├── datasets.yaml                  # Dataset definitions & download URLs
│   ├── models.yaml                    # Model configs, limits, hyperparams
│   └── experiments.yaml               # Experiment matrix definition
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                  # Unified dataset loader
│   │   ├── preprocessor.py            # Minimal preprocessing (for models that need it)
│   │   └── splitter.py                # Stratified splits, scaling experiments
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract base class for all wrappers
│   │   ├── tabpfn_wrapper.py          # TabPFN v1 / v2 / v2.5 / Real-TabPFN-2.5
│   │   ├── tabicl_wrapper.py          # TabICL v1.1 / v2
│   │   ├── mitra_wrapper.py           # Mitra via AutoGluon
│   │   ├── tabdpt_wrapper.py          # TabDPT
│   │   ├── tabnet_wrapper.py          # TabNet (pytorch-tabnet)
│   │   ├── ft_transformer_wrapper.py  # FT-Transformer
│   │   ├── gbdt_wrapper.py            # XGBoost, CatBoost, LightGBM
│   │   └── sklearn_wrapper.py         # RandomForest, LogisticRegression
│   │
│   ├── finetuning/
│   │   ├── __init__.py
│   │   ├── tabpfn_finetune.py         # TabPFN v2/v2.5 fine-tuning
│   │   ├── tabnet_finetune.py         # TabNet self-supervised → supervised
│   │   └── ft_transformer_finetune.py # FT-Transformer training
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # AUC, accuracy, log-loss, Brier, F1, calibration
│   │   ├── timing.py                  # Wall-clock fit + predict timing
│   │   ├── memory.py                  # Peak GPU/CPU memory tracking
│   │   └── statistical_tests.py       # Wilcoxon signed-rank, critical difference
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── leaderboard.py             # Generate comparison tables
│       ├── plots.py                   # Bar charts, radar plots, scaling curves
│       └── report.py                  # Auto-generate markdown report
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA on credit datasets
│   ├── 02_zero_shot_benchmark.ipynb   # Run all models on raw data
│   ├── 03_finetuning_experiments.ipynb # Fine-tuning pipeline
│   ├── 04_scaling_experiments.ipynb   # Row-scaling stress tests
│   ├── 05_analysis_and_plots.ipynb    # Final analysis
│   └── 06_quick_start_demo.ipynb      # 5-minute intro for newcomers
│
├── scripts/
│   ├── run_benchmark.py               # CLI: full benchmark pipeline
│   ├── run_scaling_test.py            # CLI: scaling experiments
│   └── generate_report.py            # CLI: produce final report
│
├── results/                           # Auto-generated
│   ├── raw_results.csv
│   ├── finetuned_results.csv
│   ├── scaling_results.csv
│   ├── figures/
│   └── report.md
│
└── tests/
    ├── test_loaders.py
    ├── test_wrappers.py
    └── test_metrics.py
```

---

## 🔬 Experimental Design

### Phase 1: Dataset Preparation

**Primary Dataset: "Give Me Some Credit" (Kaggle)**
- ~150K rows, 11 features, binary classification (default/no-default)
- Real-world credit scoring with missing values, class imbalance (~6.7% positive)
- Features: age, income, debt ratio, number of open credit lines, delinquencies, etc.

**Secondary Datasets (for robustness):**
- **German Credit (UCI)** — 1,000 rows, 20 features (tests small-data regime)
- **Taiwan Credit (UCI)** — 30,000 rows, 23 features (medium-data regime)
- **Lending Club** — 2.2M rows (tests scaling beyond TFM limits)

**Splits:**
- 60% train / 20% validation / 20% test (stratified by target)
- 5-fold cross-validation for statistical significance
- Fixed random seed (42) for reproducibility

### Phase 2: Zero-Shot / Raw Data Benchmark

Every model receives the **exact same raw data** — no feature engineering, no imputation (except for models that require it), no hyperparameter tuning.

```
For each model:
  1. Load raw train/test split
  2. Start timer
  3. model.fit(X_train, y_train)      ← For TFMs, this just stores context
  4. y_pred = model.predict_proba(X_test)
  5. Stop timer
  6. Record: AUC-ROC, Accuracy, Log-Loss, Brier Score, F1, ECE
  7. Record: fit_time, predict_time, peak_memory_mb
```

**What we measure:**
| Metric | Why It Matters |
|--------|---------------|
| AUC-ROC | Ranking quality (primary metric for credit scoring) |
| Log-Loss | Probabilistic calibration — critical for credit decisions |
| Brier Score | Proper scoring rule for probability quality |
| ECE (Expected Calibration Error) | Are predicted probabilities reliable? |
| F1 (macro) | Class-imbalanced performance |
| Accuracy | Baseline sanity check |
| Fit Time (s) | How long to "train" (or store context) |
| Predict Time (s) | Inference latency per batch |
| Peak Memory (MB) | GPU/CPU memory footprint |

### Phase 3: Fine-Tuning Experiments

Models that support fine-tuning get a dedicated experiment:

| Model | Fine-Tuning Method | Key Hyperparameters |
|-------|-------------------|---------------------|
| **TabPFN v2** | `FinetunedTabPFNClassifier` | lr=1e-5, epochs=30, batch_size=20 |
| **TabPFN v2.5** | `FinetunedTabPFNClassifier` | lr=1e-5, epochs=30, batch_size=20 |
| **TabNet** | Self-supervised pretrain → supervised | pretraining_ratio=0.8, n_d=64, n_a=64 |
| **FT-Transformer** | Standard training from scratch | lr=1e-4, epochs=200, patience=20 |
| **XGBoost (tuned)** | Optuna 100-trial HPO | n_estimators, max_depth, learning_rate, etc. |
| **CatBoost (tuned)** | Optuna 100-trial HPO | depth, learning_rate, iterations, etc. |

**We compare:**
- Zero-shot performance vs. fine-tuned performance (delta)
- Fine-tuning time investment vs. accuracy gain
- Whether fine-tuning closes the gap with tuned GBDTs

### Phase 4: Scaling Experiments

Test how each model handles increasing dataset sizes:

```
Row counts: [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 150000]

For each row count:
  1. Subsample train set (stratified)
  2. Run all applicable models
  3. Record metrics + timing + memory
  4. Note: skip models that exceed their row limit
```

**Scaling Limit Map:**
```
500 ──────────────────────────────────────────────────── 1M+
│                                                        │
│  TabPFN v1 (1K) ■                                      │
│  TabPFN v2 (10K) ■■■■■■                                │
│  Mitra (~10K) ■■■■■■                                   │
│  TabDPT (~10K) ■■■■■■                                  │
│  TabPFN v2.5 (50K) ■■■■■■■■■■■■■■■■                   │
│  TabICL v1.1 (~50K) ■■■■■■■■■■■■■■■■                  │
│  TabICLv2 (1M) ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ │
│  XGBoost/CatBoost (unlimited) ■■■■■■■■■■■■■■■■■■■■■■■ │
│  TabNet (unlimited) ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ │
```

### Phase 5: Additional Experiments

#### 5a. Missing Value Robustness
Artificially inject missing values at 5%, 10%, 20%, 30%, 50% rates and measure degradation.

#### 5b. Feature Importance / Interpretability
- TabPFN: extract attention-based feature importance
- TabNet: built-in sparse attention masks
- XGBoost/CatBoost: SHAP values
- Compare: do models agree on important features?

#### 5c. Calibration Analysis
- Plot reliability diagrams for each model
- Test whether TFMs provide better-calibrated probabilities (a claimed advantage)
- Apply Platt scaling and isotonic regression post-hoc and re-evaluate

#### 5d. Ensemble Experiments
- TabPFN + XGBoost ensemble (weighted average)
- TabICLv2 + CatBoost ensemble
- All-TFM ensemble (TabPFN + TabICL + Mitra)
- Compare: does ensembling TFMs with GBDTs yield consistent gains?

#### 5e. Preprocessing Impact
Test whether preprocessing helps or hurts TFMs:
- Raw data (baseline)
- Standard scaling + one-hot encoding
- Target encoding for categoricals
- Feature engineering (interaction terms, binning)

#### 5f. Class Imbalance Sensitivity
- Test with natural imbalance (~6.7%)
- SMOTE oversampling
- Class-weighted training
- Threshold optimization

#### 5g. Inference Latency Profiling
- Single-sample latency (simulating real-time credit decisions)
- Batch latency (1K, 10K, 50K samples)
- CPU vs GPU comparison

#### 5h. Cross-Dataset Transfer
- Train on Give Me Some Credit → test on Taiwan Credit (and vice versa)
- Tests whether TFMs generalize better than GBDTs across credit domains

---

## 🚀 Research Pipeline

### Run the Full Benchmark

```bash
# Zero-shot benchmark on all models
python scripts/run_benchmark.py --dataset give_me_credit --phase zero_shot

# Fine-tuning experiments
python scripts/run_benchmark.py --dataset give_me_credit --phase finetune

# Scaling experiments
python scripts/run_scaling_test.py --dataset give_me_credit

# Generate report
python scripts/generate_report.py
```

---

## 📋 Requirements

### Hardware
- **Minimum**: 16GB RAM, any modern CPU (runs XGBoost, TabPFN v2 on CPU)
- **Recommended**: 24GB+ RAM, NVIDIA GPU with 16GB+ VRAM (RTX 4090, A100, etc.)
- **For TabICLv2 at scale**: 80GB GPU (H100) for million-row experiments

### Software
- Python 3.10+
- CUDA 11.8+ (for GPU models)

---

## 🧠 Key Findings Template

After running, the repo auto-generates a report with these sections:

```markdown
## Leaderboard (Zero-Shot on Give Me Some Credit)
| Rank | Model          | AUC-ROC | Log-Loss | Time (s) | Memory (MB) |
|------|----------------|---------|----------|----------|-------------|
| 1    | ...            | ...     | ...      | ...      | ...         |

## Fine-Tuning Impact
| Model          | Zero-Shot AUC | Fine-Tuned AUC | Δ AUC | FT Time (s) |
|----------------|---------------|----------------|-------|-------------|

## Scaling Behavior
[Auto-generated line charts: AUC vs. row count for each model]

## Known Limitations Discovered
- TabPFN v1: crashes above 1,000 rows
- TabPFN v2.5: OOM on GPU with <16GB at 50K rows
- Mitra: degraded accuracy above 100 features
- ...
```

---

## 🤝 Contributing

We welcome contributions! Priority areas:
- Adding new TFM models as they're released
- Additional credit/financial datasets
- Improved visualization
- Cloud-based benchmarking configs (AWS, GCP)

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📖 Citation

```bibtex
@software{tfm_benchmark_2026,
  title={TFM-Bench: Benchmarking Tabular Foundation Models on Credit Data},
  author={Your Name},
  year={2026},
  url={https://github.com/YOUR_USERNAME/tfm-benchmark}
}
```

---

## 📚 References

- Hollmann et al. (2025). *Accurate predictions on small data with a tabular foundation model.* Nature.
- Grinsztajn et al. (2025). *TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models.* arXiv.
- Qu et al. (2025). *TabICL: A Tabular Foundation Model for In-Context Learning on Large Data.* ICML.
- Qu et al. (2026). *TabICLv2: A better, faster, scalable, and open tabular foundation model.* arXiv.
- Dong et al. (2025). *Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models.* NeurIPS.
- Ma et al. (2024). *TabDPT: Scaling Tabular Foundation Models on Real Data.* NeurIPS.
- Arik & Pfister (2021). *TabNet: Attentive Interpretable Tabular Learning.* AAAI.
- Gorishniy et al. (2021). *Revisiting Deep Learning Models for Tabular Data.* NeurIPS.
