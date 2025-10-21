# Home Credit — Inference Pipelines (codev1.py & codev2.py)

This README explains the purpose, inputs/outputs, dependencies, and execution steps for the two inference scripts:

- **`codev1.py`** — Polars-based feature assembly + hybrid (CatBoost + LightGBM) voting inference with **post‑blend rule**. fileciteturn0file0
- **`codev2.py`** — A leaner, fully self‑contained feature assembly + saved model loader producing a **baseline submission**; can feed into `codev1.py` for ensembling. fileciteturn0file1

---

## Quick Start

1. Place Kaggle competition data under:
   - `/kaggle/input/home-credit-credit-risk-model-stability/parquet_files/test/…`
2. Models:
   - For `codev1.py`: a pickle at `/kaggle/input/newmodel1/newmodel.pkl` containing `[cat_cols, fitted_models_cat, fitted_models_lgb]`. fileciteturn0file0
   - For `codev2.py`: joblib bundles under `/kaggle/input/homecredit-models-public/other/{lgb,cat}/1/…`. fileciteturn0file1
3. Run:
   ```bash
   python codev2.py   # produces submission_codev2.csv
   python codev1.py   # blends with submission_codev2.csv and writes submission.csv
   ```

---

## Data Layout

Both scripts expect the official parquet hierarchy (test side) under `ROOT = "/kaggle/input/home-credit-credit-risk-model-stability"`, notably:
- `parquet_files/test/test_base.parquet`
- Feature “depth” tables (`*_0.parquet`, `*_1_*.parquet`, `*_2*.parquet`) which are aggregated and joined by `case_id`. fileciteturn0file0 fileciteturn0file1

---

## What Each Script Does

### codev2.py — Baseline pipeline + saved-model inference
- **Schema & typing** via `Pipeline.set_table_dtypes(...)` for `case_id`, numeric suffix conventions (`P/A` as floats, `M` strings, `D` dates). fileciteturn0file1
- **Aggregation** (`Aggregator`) over numeric/date/string/T/L/`num_group*` columns: max/last/mean (and a few medians/vars kept internal). fileciteturn0file1
- **Feature engineering & joining**: joins depth‑0/1/2 tables on `case_id`, adds calendar features from `date_decision`, converts date diffs to day counts, drops source date columns. fileciteturn0file1
- **Model loading**: loads LGB and CatBoost model sets and associated `cols`/`cat_cols` from Joblib artifacts. fileciteturn0file1
- **Voting model**: simple mean of predicted probabilities across the LGB (first) and Cat (last) models (with Cat inputs cast to `str/category`). Writes `submission_codev2.csv`. fileciteturn0file1

### codev1.py — Extended pipeline + blend & rule
- **Utility & SchemaGen**: richer Polars helpers, schema scanning from glob patterns, optional depth‑wise groupby with max/mean/var reducers, and memory down‑casting (Polars and NumPy aware). fileciteturn0file0
- **Column filtering & transforms**:
  - Drops high‑null columns (>95%) and high‑cardinality or constant string features. fileciteturn0file0
  - Parses `riskassesment_302T` ranges into `[rng, mean]` numeric features when present. fileciteturn0file0
  - Converts date columns to day offsets relative to `date_decision`; extracts `year`, `day`, and renames `MONTH→month`, `WEEK_NUM→week_num`. fileciteturn0file0
- **Model loading**: unpickles `[cat_cols, fitted_models_cat, fitted_models_lgb]`, converts categorical columns for Cat, aligns to LGB `feature_name_`. fileciteturn0file0
- **Ensemble & post‑blend**:
  - Reads `submission_codev2.csv`, **50/50 blend** with current model’s probability. fileciteturn0file0
  - Applies a **heuristic down‑adjustment** (-0.025, clipped at 0) to a subset defined by a threshold on a specific feature column (`X_test.columns[376]`). Outputs final `submission.csv`. fileciteturn0file0

---

## Inputs & Outputs

| Script     | Inputs                                                                                          | Output file                  |
|------------|--------------------------------------------------------------------------------------------------|------------------------------|
| codev2.py  | Parquet test tables; Joblib LGB & Cat models + metadata (`cols`, `cat_cols`)                    | `submission_codev2.csv`      |
| codev1.py  | Parquet test tables; `newmodel.pkl` (Cat & LGB lists + `cat_cols`); `submission_codev2.csv`     | `submission.csv` (final)     |

---

## Dependencies

- Python 3.10+ (Kaggle Runtime OK)
- Core: `polars`, `pandas`, `numpy`, `lightgbm`, `catboost` (only in `codev1.py`), `joblib` (only in `codev2.py`)
- For speed & memory: both scripts implement type down‑casting / memory trimming (Polars in `codev1.py`, Pandas in `codev2.py`). fileciteturn0file0 fileciteturn0file1

---

## How to Run (recommended order)

1) **Generate a base submission**  
```bash
python codev2.py
# => writes submission_codev2.csv in working dir
```

2) **Blend with a stronger stack**  
```bash
python codev1.py
# => reads submission_codev2.csv and writes submission.csv
```

> Tip: If you want to run only `codev1.py` (without the 50/50 blend), remove the lines that read `submission_codev2.csv` and the blend/threshold rule. fileciteturn0file0

---

## Implementation Notes & Gotchas

- **Feature alignment**: `codev1.py` explicitly selects `X_test = X_test[fitted_models_lgb[0].feature_name_]` to match trained LGB features; ensure your pickle’s models expose `feature_name_`. fileciteturn0file0
- **Categoricals**: Before CatBoost predictions, cast `X[cat_cols]` to `str` (or `category`) as done in both scripts. Mismatched dtypes cause silent degradation. fileciteturn0file1 fileciteturn0file0
- **Date arithmetic**: Date columns ending with `D` are converted to **day deltas** from `date_decision`; verify your upstream parquet schema keeps `date_decision` present in base tables. fileciteturn0file1 fileciteturn0file0
- **Heuristic rule (codev1)**: The down‑adjustment by `-0.025` for a subset defined by column index `376` is dataset‑specific; re‑validate this threshold on CV or OOF to avoid leaderboard overfit. fileciteturn0file0

---

## Extending the Pipelines

- Add/remove aggregation expressions in `Aggregator` to tune feature richness vs. RAM/CPU. fileciteturn0file1 fileciteturn0file0
- Tighten column filters (null‑ratio / cardinality) and memory shims for large‑RAM instances.
- Swap the final **mean** voting with weighted voting or rank‑averaging if model scales differ.

---

## Repro Checklist

- [ ] Parquet test files present and readable
- [ ] Model artifacts available in expected paths
- [ ] `submission_codev2.csv` successfully produced before running `codev1.py`
- [ ] Same Python/package versions as training (LightGBM/CatBoost) to ensure `feature_name_` compatibility

---

*Maintainers:* Dx & team  
*Last updated:* auto‑generated from source context.
