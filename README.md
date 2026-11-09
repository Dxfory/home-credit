# Home Credit — Inference Pipelines (codev1.py & codev2.py)

[![CI](https://github.com/Dxfory/home-credit/actions/workflows/ci.yml/badge.svg)](https://github.com/Dxfory/home-credit/actions/workflows/ci.yml)

**Inference pipeline for the team gold, 9th of 3,856. `codev2.py` writes the baseline submission. `codev1.py` blends it 50/50 and applies the column-376 stability adjustment. Model weights and parquet files come from the Kaggle competition inputs.**

Team gold, Home Credit — Credit Risk Model Stability, certificate dated 2024-05-28, issued to **X. Dxfory**. The public pipeline assembles depth-0/1/2 features in Polars, runs LightGBM and CatBoost voting, and applies a 50/50 blend plus a stability adjustment. Time-aware grouped cross-validation and the hand-built features are the training design behind that gold-medal inference stack.

Scripts live in `home_credit/`. `home_credit/codev1.py` reads parquet from `../input/home-credit-credit-risk-model-stability`. `home_credit/codev2.py` reads parquet from `/kaggle/input/home-credit-credit-risk-model-stability`. On Kaggle, run from the working directory that matches those roots. CI runs `python -m py_compile` on both scripts. Competition parquet and model weights stay with the Kaggle competition inputs.

---

## Quick Start

1. Place Kaggle competition data so each script's `ROOT` resolves:
   - `codev2.py`: `/kaggle/input/home-credit-credit-risk-model-stability/parquet_files/test/…`
   - `codev1.py`: `../input/home-credit-credit-risk-model-stability/parquet_files/test/…` relative to the working directory
2. Models:
   - For `codev1.py`: a pickle at `/kaggle/input/newmodel1/newmodel.pkl` containing `[cat_cols, fitted_models_cat, fitted_models_lgb]`.
   - For `codev2.py`: joblib bundles under `/kaggle/input/homecredit-models-public/other/{lgb,cat}/1/…`.
3. Run, from the directory that matches those roots:
   ```bash
   python home_credit/codev2.py   # produces submission_codev2.csv
   python home_credit/codev1.py   # blends with submission_codev2.csv and writes submission.csv
   ```

---

## Data Layout

`codev1.py` sets `ROOT = Path("../input/home-credit-credit-risk-model-stability")`. `codev2.py` sets `ROOT = "/kaggle/input/home-credit-credit-risk-model-stability"`. Both expect the official parquet hierarchy (test side), notably:

- `parquet_files/test/test_base.parquet`
- Feature “depth” tables (`*_0.parquet`, `*_1_*.parquet`, `*_2*.parquet`) which are aggregated and joined by `case_id`.

---

## What Each Script Does

### codev2.py — Baseline pipeline + saved-model inference
- **Schema & typing** via `Pipeline.set_table_dtypes(...)` for `case_id`, numeric suffix conventions (`P/A` as floats, `M` strings, `D` dates).
- **Aggregation** (`Aggregator`) over numeric/date/string/T/L/`num_group*` columns: max/last/mean (and a few medians/vars kept internal).
- **Feature engineering & joining**: joins depth‑0/1/2 tables on `case_id`, adds calendar features from `date_decision`, converts date diffs to day counts, drops source date columns.
- **Model loading**: loads LGB and CatBoost model sets and associated `cols`/`cat_cols` from Joblib artifacts.
- **Voting model**: simple mean of predicted probabilities across the LGB (first) and Cat (last) models (with Cat inputs cast to `str/category`). Writes `submission_codev2.csv`.

### codev1.py — Extended pipeline + blend & rule
- **Utility & SchemaGen**: richer Polars helpers, schema scanning from glob patterns, optional depth‑wise groupby with max/mean/var reducers, and memory down‑casting (Polars and NumPy aware).
- **Column filtering & transforms**:
  - Drops high‑null columns (>95%) and high‑cardinality or constant string features.
  - Parses `riskassesment_302T` ranges into `[rng, mean]` numeric features when present.
  - Converts date columns to day offsets relative to `date_decision`; extracts `year`, `day`, and renames `MONTH→month`, `WEEK_NUM→week_num`.
- **Model loading**: unpickles `[cat_cols, fitted_models_cat, fitted_models_lgb]`, converts categorical columns for Cat, aligns to LGB `feature_name_`.
- **One predict path, then ensemble & post‑blend**:
  - Loads `newmodel.pkl`, aligns columns to `feature_name_`, and scores once.
  - Reads `submission_codev2.csv`, **50/50 blend** with the current model’s probability.
  - Applies a **stability adjustment** (−0.025, floored at 0) on the subset defined by **column 376** of that aligned frame (`X_test.columns[376]`). Writes `submission.csv`.

---

## Inputs & Outputs

| Script     | Inputs                                                                                          | Output file                  |
|------------|--------------------------------------------------------------------------------------------------|------------------------------|
| codev2.py  | Parquet test tables; Joblib LGB & Cat models + metadata (`cols`, `cat_cols`)                    | `submission_codev2.csv`      |
| codev1.py  | Parquet test tables; `newmodel.pkl` (Cat & LGB lists + `cat_cols`); `submission_codev2.csv`     | `submission.csv` (final)     |

---

## Dependencies

- Python 3.10+ (Kaggle runtime)
- `requirements.txt`: `polars`, `pandas`, `numpy`, `lightgbm`, `catboost`, `joblib`, `scikit-learn`
- `codev1.py` also imports `IPython.display`, which the Kaggle runtime provides
- Both scripts implement type down‑casting / memory trimming (Polars in `codev1.py`, Pandas in `codev2.py`)

---

## How to Run (recommended order)

1) **Generate a base submission**
```bash
python home_credit/codev2.py
# => writes submission_codev2.csv in the working directory
```

2) **Blend with the codev1 stack**
```bash
python home_credit/codev1.py
# => reads submission_codev2.csv and writes submission.csv
```

The blend in `codev1.py` reads `submission_codev2.csv`, so the baseline file is present before that step. Column 376 is the stability adjustment on the blended score.

---

## Implementation Notes & Gotchas

- **Feature alignment**: `codev1.py` explicitly selects `X_test = X_test[fitted_models_lgb[0].feature_name_]` to match trained LGB features; the pickle’s models expose `feature_name_`.
- **Categoricals**: Before CatBoost predictions, cast `X[cat_cols]` to `str` (or `category`) as done in both scripts. Matching dtypes keeps the vote on the trained columns.
- **Date arithmetic**: Date columns ending with `D` are converted to **day deltas** from `date_decision`; the upstream parquet schema keeps `date_decision` present in base tables.
- **Stability adjustment (codev1)**: The −0.025 adjustment, floored at 0, applies to the subset defined by column index `376` after alignment to `feature_name_`. That index is the competition feature layout behind the team-gold submission.

---

## Extending the Pipelines

- Add or remove aggregation expressions in `Aggregator` to tune feature richness against RAM and CPU.
- Tighten column filters (null‑ratio / cardinality) and memory shims for large‑RAM instances.
- Swap the final **mean** voting with weighted voting or rank‑averaging when model scales differ.

---

## Repro Checklist

- [ ] Parquet test files present and readable at the script `ROOT`
- [ ] Model artifacts available in the expected paths
- [ ] `submission_codev2.csv` written before `codev1.py`
- [ ] Same Python and package versions as training (LightGBM/CatBoost) so `feature_name_` matches

---

*Maintainers:* Dx & team

License: MIT
