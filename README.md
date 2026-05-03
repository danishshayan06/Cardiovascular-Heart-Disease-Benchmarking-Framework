# Cardiovascular Benchmarking Framework
### A Comparative Implementation of FT-Transformers and Ensemble Models

This framework provides a unified pipeline for benchmarking the **FT-Transformer** architecture (Gorishniy et al., 2021) against traditional machine learning baselines using clinical cardiovascular data. It is specifically designed to handle multi-dataset evaluation across different geographic subsets of the UCI Heart Disease repository.

## 📂 Project Structure
To ensure the script executes correctly, maintain the following directory structure:
```text
/project-root
├── benchmark_3.py       # Main execution script
├── cleveland.csv        # Primary Dataset
├── hungarian.csv        # Supplementary Dataset
├── switzerland.csv      # Supplementary Dataset
└── README.md            # Documentation
```

## 🚀 Installation & Setup
1. **Environment:** It is recommended to use the `rtdl_project` Conda environment created during Phase 1.
2. **Dependencies:** Ensure the following libraries are installed:
   * `torch`, `numpy`, `pandas`, `scikit-learn`
   * `xgboost`, `lightgbm` (optional but recommended for baselines)

## 💻 Usage Instructions

### **Basic Execution**
To run the benchmark on a single local dataset:
```bash
python benchmark_3.py --data cleveland.csv
```

### **Multi-Dataset Benchmarking**
To perform a comparative analysis across multiple datasets simultaneously:
```bash
python benchmark_3.py --data cleveland.csv hungarian.csv switzerland.csv
```

### **Command Line Arguments**
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--data` | **Required** | Space-separated list of CSV file paths. |
| `--target` | `target` | The name of the label column in your CSV. |
| `--epochs` | `50` | Number of training iterations for Deep Learning models. |
| `--batch` | `256` | Batch size for training. |
| `--outdir` | `results` | Folder where all reports and CSVs will be saved. |

## 📊 Outputs & Evaluation
The framework generates 10 distinct metrics per model to ensure rigorous experimental analysis:

*   **Clinical Accuracy:** Sensitivity (Recall), Specificity, and Confusion Matrix (TN, FP, FN, TP).
*   **Statistical Metrics:** Accuracy, Precision, and F1-Score.
*   **Probabilistic Metrics:** ROC-AUC, PR-AUC, and Log Loss.

### **Generated Files (in `/results`):**
*   **`[dataset]_results.csv`**: Detailed metrics for each individual dataset.
*   **`cross_reference.csv`**: A master comparison of all models across all tested datasets.
*   **`average_scores.csv`**: Macro-averaged performance scores used for the final technical report.

## 📝 Academic Implementation Details
*   **Preprocessing:** Includes median imputer for numerical data, mode imputer for categorical data, and standard scaling fit strictly on the training split (70/15/15).
*   **Model Architecture:** Implements the **Feature Tokenizer** and **Transformer Encoder** as described in the NeurIPS 2021 paper by Gorishniy et al.
*   **Reproducibility:** Seeded with `SEED = 42` to ensure consistent results across different runs.

---
**Course:** AI2002 — Artificial Intelligence  
**Institution:** FAST NUCES, Lahore Campus  
**Submission Date:** May 3, 2026