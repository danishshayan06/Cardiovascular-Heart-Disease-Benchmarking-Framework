import argparse
import warnings
import math
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── CLI ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Cardiovascular Benchmarking Framework — Multi-Dataset",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument(
    "--data", nargs="+", required=True,
    metavar="FILE",
    help="One or more CSV dataset paths.\n"
         "Example: --data cleveland.csv hungarian.csv"
)
parser.add_argument(
    "--target", type=str, default="target",
    help="Target column name (default: 'target')"
)
parser.add_argument("--epochs",  type=int,   default=50)
parser.add_argument("--lr",      type=float, default=1e-4)
parser.add_argument("--batch",   type=int,   default=256)
parser.add_argument("--no-deep", action="store_true",
                    help="Skip FT-Transformer and ResNet")
parser.add_argument("--device",  type=str,   default="auto",
                    help="cpu / cuda / auto")
parser.add_argument("--outdir",  type=str,   default="results",
                    help="Directory to save all output CSVs (default: results/)")
args = parser.parse_args()

# ── Output directory ─────────────────────────────────────────
OUT_DIR = Path(args.outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1.  IMPORTS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("   Cardiovascular Benchmarking Framework  —  Multi-Dataset Mode")
print("   FT-Transformer vs Ensemble Models  |  UCI Heart Disease")
print("=" * 72)
print(f"   Datasets   : {len(args.data)}")
for p in args.data:
    print(f"               {p}")
print(f"   Output dir : {OUT_DIR}/")
print("=" * 72)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.impute          import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    log_loss, confusion_matrix
)

try:
    import xgboost as xgb;    HAS_XGB = True
except ImportError:
    print("  [WARN] xgboost not installed — XGBoost will be skipped")
    HAS_XGB = False

try:
    import lightgbm as lgb;   HAS_LGB = True
except ImportError:
    print("  [WARN] lightgbm not installed — LightGBM will be skipped")
    HAS_LGB = False

HAS_TORCH = False
if not args.no_deep:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        HAS_TORCH = True
        DEVICE = torch.device(
            "cuda" if (args.device == "auto" and torch.cuda.is_available())
            else (args.device if args.device != "auto" else "cpu")
        )
        print(f"  PyTorch device : {DEVICE}")
    except ImportError:
        print("  [WARN] PyTorch not installed — deep models will be skipped")


# ─────────────────────────────────────────────────────────────
# 2.  DATASET SCHEMA
# ─────────────────────────────────────────────────────────────
UCI_COLUMNS      = ["age","sex","cp","trestbps","chol","fbs",
                    "restecg","thalach","exang","oldpeak","slope","ca","thal","target"]
CATEGORICAL_COLS = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]
NUMERICAL_COLS   = ["age","trestbps","chol","thalach","oldpeak"]
METRIC_COLS      = ["Accuracy","Precision","Recall","F1",
                    "ROC-AUC","PR-AUC","Log Loss","Sensitivity","Specificity"]
KEY_METRICS      = ["Accuracy","F1","ROC-AUC","PR-AUC","Sensitivity","Specificity"]


# ─────────────────────────────────────────────────────────────
# 3.  DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────
def load_dataset(path: str) -> pd.DataFrame:

    path = str(path)
    if not os.path.isfile(path):
        print(f"  [ERROR] File not found: {path}")
        sys.exit(1)

    peek = pd.read_csv(path, nrows=1, header=None)
    has_header = not pd.to_numeric(peek.iloc[0, 0], errors="coerce") == peek.iloc[0, 0]

    df = pd.read_csv(
        path,
        header=0 if has_header else None,
        na_values=["?"]
    )

    
    if df.shape[1] == 14 and not has_header:
        df.columns = UCI_COLUMNS
    elif df.shape[1] == 14 and has_header:
        
        cols = list(df.columns)
        if cols[-1] != "target":
            cols[-1] = "target"
            df.columns = cols
    elif "target" not in df.columns and args.target in df.columns:
        df = df.rename(columns={args.target: "target"})
    elif "target" not in df.columns:
        print(f"  [ERROR] Cannot find target column in {path}.")
        print(f"          Pass --target <colname> to specify it.")
        sys.exit(1)

    df["target"] = (pd.to_numeric(df["target"], errors="coerce") > 0).astype(int)
    return df


def preprocess(df: pd.DataFrame):

    num_cols = [c for c in NUMERICAL_COLS   if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    all_feat  = num_cols + cat_cols

    X = df[all_feat].copy()
    y = df["target"].values

    if num_cols:
        X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])
    if cat_cols:
        X[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])
        X[cat_cols] = X[cat_cols].astype(int)
        le = LabelEncoder()
        for col in cat_cols:
            X[col] = le.fit_transform(X[col])

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)

    
    if num_cols:
        sc = StandardScaler()
        X_tr[num_cols] = sc.fit_transform(X_tr[num_cols])
        X_va[num_cols] = sc.transform(X_va[num_cols])
        X_te[num_cols] = sc.transform(X_te[num_cols])

    cat_cards = [int(X[c].nunique()) for c in cat_cols] if cat_cols else []
    return (X_tr.values, y_tr,
            X_va.values, y_va,
            X_te.values, y_te,
            len(num_cols), cat_cards)


# ─────────────────────────────────────────────────────────────
# 4.  EVALUATION — 10 METRICS
# ─────────────────────────────────────────────────────────────
def full_evaluate(model_name: str, dataset_name: str,
                  y_true, y_pred, y_prob) -> dict:
    
    cm             = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy    = accuracy_score(y_true, y_pred)
    precision   = precision_score(y_true, y_pred, zero_division=0)
    recall      = recall_score(y_true, y_pred, zero_division=0)
    f1          = f1_score(y_true, y_pred, zero_division=0)
    roc_auc     = roc_auc_score(y_true, y_prob)
    pr_auc      = average_precision_score(y_true, y_prob)
    logloss     = log_loss(y_true, y_prob)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    tag     = "  [PROPOSED MODEL]" if model_name == "FT-Transformer" else "  [BASELINE]      "
    divider = "-" * 62

    print("\n" + divider)
    print(f"  Model       : {model_name}{tag}")
    print(f"  Dataset     : {dataset_name}")
    print(divider)
    print(f"  Accuracy    : {accuracy:.4f}")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Recall      : {recall:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  ROC-AUC     : {roc_auc:.4f}")
    print(f"  PR-AUC      : {pr_auc:.4f}")
    print(f"  Log Loss    : {logloss:.4f}  (lower = better)")
    print(f"  Sensitivity : {sensitivity:.4f}  (TP / (TP+FN))")
    print(f"  Specificity : {specificity:.4f}  (TN / (TN+FP))")
    print(f"  Confusion Matrix:")
    print(f"                   Pred: No Disease    Pred: Disease")
    print(f"    True: No Dis       {tn:<10}         {fp:<10}")
    print(f"    True: Disease      {fn:<10}         {tp:<10}")
    print(divider)

    return {
        "Dataset":     dataset_name,
        "Model":       model_name,
        "Accuracy":    accuracy,
        "Precision":   precision,
        "Recall":      recall,
        "F1":          f1,
        "ROC-AUC":     roc_auc,
        "PR-AUC":      pr_auc,
        "Log Loss":    logloss,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


# ─────────────────────────────────────────────────────────────
# 5.  DEEP LEARNING MODELS
# ─────────────────────────────────────────────────────────────
if HAS_TORCH:

    class FeatureTokenizer(nn.Module):
        def __init__(self, n_num, cat_cards, d):
            super().__init__()
            self.n_num = n_num
            if n_num:
                self.w = nn.Parameter(torch.empty(n_num, d))
                self.b = nn.Parameter(torch.zeros(n_num, d))
                nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
            if cat_cards:
                self.embs = nn.ModuleList(
                    [nn.Embedding(c + 1, d) for c in cat_cards])

        def forward(self, x):
            parts = []
            if self.n_num:
                parts.append(x[:, :self.n_num].unsqueeze(-1) * self.w + self.b)
            if hasattr(self, "embs"):
                cats = x[:, self.n_num:].long()
                parts.append(torch.stack(
                    [e(cats[:, i]) for i, e in enumerate(self.embs)], dim=1))
            return torch.cat(parts, dim=1)

    class FTTransformer(nn.Module):
        """Gorishniy et al. (2021) — Feature Tokenizer + Transformer."""
        def __init__(self, n_num, cat_cards,
                     d=192, heads=8, layers=3, ffn_factor=4/3, dropout=0.2):
            super().__init__()
            self.tok  = FeatureTokenizer(n_num, cat_cards, d)
            self.cls  = nn.Parameter(torch.zeros(1, 1, d))
            enc = nn.TransformerEncoderLayer(
                d_model=d, nhead=heads,
                dim_feedforward=int(d * ffn_factor),
                dropout=dropout, activation="relu",
                batch_first=True, norm_first=True)
            self.tf   = nn.TransformerEncoder(enc, num_layers=layers)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 2))

        def forward(self, x):
            t = torch.cat([self.cls.expand(x.size(0), -1, -1), self.tok(x)], dim=1)
            return self.head(self.tf(t)[:, 0])

    class ResBlock(nn.Module):
        def __init__(self, d, p):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(d), nn.Linear(d, d*2), nn.ReLU(),
                nn.Dropout(p),   nn.Linear(d*2, d), nn.Dropout(p))
        def forward(self, x): return x + self.net(x)

    class ResNet(nn.Module):
        """ResNet-style tabular model (Gorishniy et al., 2021)."""
        def __init__(self, n_feat, d=256, blocks=4, dropout=0.2):
            super().__init__()
            self.proj   = nn.Linear(n_feat, d)
            self.blocks = nn.Sequential(*[ResBlock(d, dropout) for _ in range(blocks)])
            self.head   = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 2))
        def forward(self, x): return self.head(self.blocks(self.proj(x)))

    def _make_loader(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle)

    def train_deep(model, Xtr, ytr, Xva, yva, label):
        model.to(DEVICE)
        opt  = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        crit = nn.CrossEntropyLoss()
        tr_ld, va_ld = _make_loader(Xtr, ytr, True), _make_loader(Xva, yva, False)
        best_loss, best_state, pat = float("inf"), None, 0

        print(f"\n  Training {label} ...")
        for ep in range(1, args.epochs + 1):
            model.train()
            for xb, yb in tr_ld:
                opt.zero_grad()
                loss = crit(model(xb.to(DEVICE)), yb.to(DEVICE))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()

            model.eval(); vl = 0.0
            with torch.no_grad():
                for xb, yb in va_ld:
                    vl += crit(model(xb.to(DEVICE)), yb.to(DEVICE)).item()
            vl /= len(va_ld)
            if ep % 10 == 0:
                print(f"    epoch {ep:>3}/{args.epochs}  val_loss={vl:.4f}")
            if vl < best_loss:
                best_loss  = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                pat = 0
            else:
                pat += 1
                if pat >= 10:
                    print(f"    Early stop @ epoch {ep}")
                    break
        model.load_state_dict(best_state)
        return model

    def deep_predict(model, X):
        model.eval()
        dl = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=args.batch, shuffle=False)
        probs = []
        with torch.no_grad():
            for (xb,) in dl:
                p = torch.softmax(model(xb.to(DEVICE)), dim=1)[:, 1]
                probs.append(p.cpu().numpy())
        probs = np.concatenate(probs)
        return (probs >= 0.5).astype(int), probs


# ─────────────────────────────────────────────────────────────
# 6.  PER-DATASET TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────
def run_one_dataset(csv_path: str) -> list:

    ds_name = Path(csv_path).stem         
    W, NW   = 11, 22
    total_w = NW + W * len(METRIC_COLS)

    # ── Banner ───────────────────────────────────────────────
    print("\n\n" + "#" * 72)
    print(f"#  DATASET: {ds_name.upper():<58}#")
    print(f"#  File   : {csv_path:<58}#")
    print("#" * 72)

    # ── Load ─────────────────────────────────────────────────
    df = load_dataset(csv_path)
    print(f"\n  Rows x Cols   : {df.shape}")
    print(f"  Class balance : {df['target'].value_counts().to_dict()}")
    print(f"  Missing values: {df.isnull().sum().sum()}")

    (X_train, y_train, X_val, y_val,
     X_test,  y_test,  n_num, cat_cards) = preprocess(df)
    n_feat = X_train.shape[1]

    print(f"  Train/Val/Test: {len(X_train)} / {len(X_val)} / {len(X_test)}")

    # ── Train & evaluate all models ───────────────────────────
    print(f"\n--- Training models on [{ds_name}] ---")
    results = []

    def ev(name, pred, prob):
        results.append(full_evaluate(name, ds_name, y_test, pred, prob))

    # FT-Transformer
    if HAS_TORCH:
        torch.manual_seed(SEED)
        ft = FTTransformer(n_num=n_num, cat_cards=cat_cards)
        ft = train_deep(ft, X_train, y_train, X_val, y_val, "FT-Transformer")
        pred, prob = deep_predict(ft, X_test);  ev("FT-Transformer", pred, prob)

    # ResNet
    if HAS_TORCH:
        torch.manual_seed(SEED)
        rn = ResNet(n_feat=n_feat)
        rn = train_deep(rn, X_train, y_train, X_val, y_val, "ResNet")
        pred, prob = deep_predict(rn, X_test);  ev("ResNet", pred, prob)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                 random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    ev("Random Forest", rf.predict(X_test), rf.predict_proba(X_test)[:, 1])

    # XGBoost
    if HAS_XGB:
        xgb_m = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=SEED, verbosity=0)
        xgb_m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        ev("XGBoost", xgb_m.predict(X_test), xgb_m.predict_proba(X_test)[:, 1])

    # LightGBM
    if HAS_LGB:
        lgb_m = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbose=-1)
        lgb_m.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(20, verbose=False),
                             lgb.log_evaluation(-1)])
        ev("LightGBM", lgb_m.predict(X_test), lgb_m.predict_proba(X_test)[:, 1])

    # Logistic Regression
    lr_m = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    lr_m.fit(X_train, y_train)
    ev("Logistic Regression", lr_m.predict(X_test), lr_m.predict_proba(X_test)[:, 1])

    # ── Per-dataset master table ──────────────────────────────
    df_res = pd.DataFrame(results).set_index("Model")

    print(f"\n\n{'=' * total_w}")
    print(f"  RESULTS — {ds_name.upper()}")
    print(f"  Proposed: FT-Transformer  |  All others: Baselines")
    print(f"{'=' * total_w}")
    print(f"  {'Model':<{NW}}" + "".join(f"{m:>{W}}" for m in METRIC_COLS))
    print(f"  {'-' * (total_w - 2)}")

    for name, row in df_res[METRIC_COLS].iterrows():
        marker = "* " if name == "FT-Transformer" else "  "
        vals   = "".join(f"{row[m]:>{W}.4f}" for m in METRIC_COLS)
        print(f"{marker} {name:<{NW}}{vals}")

    print(f"  {'-' * (total_w - 2)}")
    print("  * = Proposed model (FT-Transformer)\n")

    # best per metric
    print("  Best per metric:")
    for m in METRIC_COLS:
        if m == "Log Loss":
            best, val = df_res[m].idxmin(), df_res[m].min()
            note = "(lower is better)"
        else:
            best, val = df_res[m].idxmax(), df_res[m].max()
            note = ""
        star = " <-- PROPOSED" if best == "FT-Transformer" else ""
        print(f"    {m:<14}  {best:<22}  {val:.4f}  {note}{star}")
    print(f"{'=' * total_w}")

    # ── FT-Transformer vs baselines delta ────────────────────
    KEY = KEY_METRICS
    if "FT-Transformer" in df_res.index:
        ft_row    = df_res.loc["FT-Transformer"]
        baselines = [r for r in df_res.index if r != "FT-Transformer"]
        kw        = NW + W * len(KEY)
        print(f"\n{'=' * kw}")
        print(f"  FT-Transformer vs Baselines  [{ds_name}]")
        print(f"  Delta = FT - Baseline  |  (+) FT wins  |  (-) Baseline wins")
        print(f"{'=' * kw}")
        print(f"  {'Baseline':<{NW}}" + "".join(f"{m:>{W}}" for m in KEY))
        print(f"  {'-' * (kw - 2)}")
        for bl in baselines:
            bl_row = df_res.loc[bl]
            deltas = []
            for m in KEY:
                d    = ft_row[m] - bl_row[m]
                sign = "+" if d >= 0 else ""
                deltas.append(f"{sign}{d:.4f}".rjust(W))
            print(f"  {bl:<{NW}}" + "".join(deltas))
        print(f"{'=' * kw}")
        wins = sum(1 for m in KEY
                   if df_res.loc["FT-Transformer", m] >= df_res[m].max())
        print(f"\n  FT-Transformer leads on {wins}/{len(KEY)} key metrics on [{ds_name}].")

    # ── Save individual CSV ───────────────────────────────────
    save_cols = METRIC_COLS + ["TN","FP","FN","TP"]
    out_path  = OUT_DIR / f"{ds_name}_results.csv"
    df_res[save_cols].to_csv(out_path)
    print(f"\n  Saved -> {out_path}")

    return results


# ─────────────────────────────────────────────────────────────
# 7.  MAIN LOOP — ALL DATASETS
# ─────────────────────────────────────────────────────────────
all_results = []

for csv_path in args.data:
    dataset_results = run_one_dataset(csv_path)
    all_results.extend(dataset_results)


# ─────────────────────────────────────────────────────────────
# 8.  CROSS-DATASET SUMMARY  (only if > 1 dataset)
# ─────────────────────────────────────────────────────────────
if len(args.data) > 1:
    df_all    = pd.DataFrame(all_results)
    datasets  = df_all["Dataset"].unique().tolist()
    models    = df_all["Model"].unique().tolist()
    W, NW, DW = 11, 22, 16
    total_w   = NW + DW * len(datasets)

    print("\n\n" + "=" * 72)
    print("  CROSS-DATASET SUMMARY")
    print("  Proposed: FT-Transformer  |  All others: Baselines")
    print("=" * 72)

    for metric in METRIC_COLS:
        lower_is_better = (metric == "Log Loss")

        print(f"\n  Metric: {metric}  {'(lower is better)' if lower_is_better else ''}")
        print(f"  {'-' * (NW + DW * len(datasets))}")
        # header
        print(f"  {'Model':<{NW}}" +
              "".join(f"{ds:>{DW}}" for ds in datasets))
        print(f"  {'-' * (NW + DW * len(datasets))}")

        for model in models:
            row_vals = []
            for ds in datasets:
                mask = (df_all["Dataset"] == ds) & (df_all["Model"] == model)
                subset = df_all.loc[mask, metric]
                val = subset.values[0] if len(subset) else float("nan")
                row_vals.append(val)

            marker = "* " if model == "FT-Transformer" else "  "
            vals_str = "".join(
                f"{v:>{DW}.4f}" if not np.isnan(v) else f"{'N/A':>{DW}}"
                for v in row_vals
            )
            print(f"{marker} {model:<{NW}}{vals_str}")

        # best per dataset for this metric
        print(f"  {'':─<{NW + DW * len(datasets)}}")
        best_vals = []
        for ds in datasets:
            mask   = df_all["Dataset"] == ds
            subset = df_all.loc[mask, ["Model", metric]].dropna()
            if lower_is_better:
                best_m = subset.loc[subset[metric].idxmin(), "Model"]
                best_v = subset[metric].min()
            else:
                best_m = subset.loc[subset[metric].idxmax(), "Model"]
                best_v = subset[metric].max()
            best_vals.append(f"{best_m[:10]}({best_v:.3f})")
        print(f"  {'Best':<{NW}}" +
              "".join(f"{b:>{DW}}" for b in best_vals))

    # ── Average rank per model across all datasets & metrics ──
    print(f"\n\n{'=' * 72}")
    print("  AVERAGE METRIC SCORES ACROSS ALL DATASETS  (macro-average)")
    print(f"{'=' * 72}")
    W2 = 11
    print(f"  {'Model':<{NW}}" + "".join(f"{m:>{W2}}" for m in METRIC_COLS))
    print(f"  {'-' * (NW + W2 * len(METRIC_COLS))}")

    for model in models:
        avgs = []
        for metric in METRIC_COLS:
            vals = df_all.loc[df_all["Model"] == model, metric].dropna()
            avgs.append(vals.mean() if len(vals) else float("nan"))
        marker = "* " if model == "FT-Transformer" else "  "
        vals_str = "".join(
            f"{v:>{W2}.4f}" if not np.isnan(v) else f"{'N/A':>{W2}}"
            for v in avgs
        )
        print(f"{marker} {model:<{NW}}{vals_str}")
    print(f"{'=' * 72}")
    print("  * = Proposed model (FT-Transformer)")

    # ── FT-Transformer wins summary across datasets ────────────
    print(f"\n{'=' * 72}")
    print("  FT-TRANSFORMER WIN SUMMARY ACROSS ALL DATASETS")
    print(f"  (How often FT-Transformer leads on each key metric)")
    print(f"{'=' * 72}")
    print(f"  {'Metric':<16}  {'FT Wins':>8}  {'Total':>7}  {'Win Rate':>10}")
    print(f"  {'-' * 46}")
    for metric in KEY_METRICS:
        wins  = 0
        total = 0
        for ds in datasets:
            ds_mask = df_all["Dataset"] == ds
            ft_mask = df_all["Model"]   == "FT-Transformer"
            ft_vals = df_all.loc[ds_mask & ft_mask, metric]
            if len(ft_vals) == 0:
                continue
            ft_val  = ft_vals.values[0]
            best_v  = df_all.loc[ds_mask, metric].max()
            total  += 1
            if ft_val >= best_v:
                wins += 1
        rate = wins / total if total > 0 else 0.0
        print(f"  {metric:<16}  {wins:>8}  {total:>7}  {rate:>9.1%}")
    print(f"{'=' * 72}")


# ─────────────────────────────────────────────────────────────
# 9.  SAVE CROSS-REFERENCE CSVs
# ─────────────────────────────────────────────────────────────
df_all = pd.DataFrame(all_results)

# ── cross_reference.csv — all rows (dataset × model × all metrics) ──
cr_path = OUT_DIR / "cross_reference.csv"
df_all.to_csv(cr_path, index=False)
print(f"\n  Saved -> {cr_path}  ({len(df_all)} rows)")

# ── cross_summary.csv — best model per metric per dataset ──────────
summary_rows = []
for ds in df_all["Dataset"].unique():
    ds_df = df_all[df_all["Dataset"] == ds]
    row   = {"Dataset": ds}
    for metric in METRIC_COLS:
        if metric == "Log Loss":
            idx = ds_df[metric].idxmin()
        else:
            idx = ds_df[metric].idxmax()
        row[f"Best_{metric}_Model"] = ds_df.loc[idx, "Model"]
        row[f"Best_{metric}_Value"] = round(ds_df.loc[idx, metric], 4)
    summary_rows.append(row)

cs_path = OUT_DIR / "cross_summary.csv"
pd.DataFrame(summary_rows).to_csv(cs_path, index=False)
print(f"  Saved -> {cs_path}")

# ── average_scores.csv — macro-average per model across all datasets ─
avg_rows = []
for model in df_all["Model"].unique():
    m_df = df_all[df_all["Model"] == model][METRIC_COLS]
    avg  = m_df.mean().round(4).to_dict()
    avg["Model"] = model
    avg_rows.append(avg)

avg_path = OUT_DIR / "average_scores.csv"
pd.DataFrame(avg_rows)[["Model"] + METRIC_COLS].to_csv(avg_path, index=False)
print(f"  Saved -> {avg_path}")

print(f"\n{'=' * 72}")
print(f"  ALL DONE.  Output files in: {OUT_DIR}/")
print(f"    Per-dataset : {'  '.join(Path(p).stem + '_results.csv' for p in args.data)}")
print(f"    Cross-ref   : cross_reference.csv")
print(f"    Summary     : cross_summary.csv")
print(f"    Avg scores  : average_scores.csv")
print(f"{'=' * 72}\n")