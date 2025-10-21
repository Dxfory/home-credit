# %%writefile codev1.py
import gc
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import warnings
import pickle
from catboost import CatBoostClassifier, Pool
from glob import glob
from IPython.display import display
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from typing import Any

warnings.filterwarnings("ignore")

ROOT = Path("../input/home-credit-credit-risk-model-stability")
TRAIN_DIR = ROOT / "parquet_files" / "train"
TEST_DIR = ROOT / "parquet_files" / "test"


class Utility:
    @staticmethod
    def get_feat_defs(ending_with: str) -> None:
        """
        Retrieves feature definitions from a CSV file based on the specified ending.
        Returns:
        - pl.DataFrame: Filtered feature definitions.
        """
        feat_defs: pl.DataFrame = pl.read_csv(ROOT / "feature_definitions.csv")

        filtered_feats: pl.DataFrame = feat_defs.filter(
            pl.col("Variable").apply(lambda var: var.endswith(ending_with))
        )

        with pl.Config(fmt_str_lengths=200, tbl_rows=-1):
            print(filtered_feats)
        filtered_feats = None
        feat_defs = None

    #     @staticmethod
    #     def find_index(lst: list[Any], item: Any) -> int | None:
    #         """
    #         Finds the index of an item)
    #         except ValueError:
    #             return None
    @staticmethod
    def find_index(lst: list[Any], item: Any) -> int | None:
        """
        Finds the index of an item in a list.

        Args:
        - lst (list): List to search.
        - item (Any): Item to find in the list.

        Returns:
        - int | None: Index of the item if found, otherwise None.
        """
        try:
            return lst.index(item)
        except ValueError:
            return None

    @staticmethod
    def dtype_to_str(dtype: pl.DataType) -> str:
        """
        Converts Polars data type to string representation.

        Args:
        - dtype (pl.DataType): Polars data type.

        Returns:
        Args:
        - dtype (pl.DataType): Polars data type.

        Returns:
        - str: String representation of the data type.
        """
        dtype_map = {
            pl.Decimal: "Decimal",
            pl.Float32: "Float32",
            pl.Float64: "Float64",
            pl.UInt8: "UInt8",
            pl.UInt16: "UInt16",
            pl.UInt32: "UInt32",
            pl.UInt64: "UInt64",
            pl.Int8: "Int8",
            pl.Int16: "Int16",
            pl.Int32: "Int32",
            pl.Int64: "Int64",
            pl.Date: "Date",
            pl.Datetime: "Datetime",
            pl.Duration: "Duration",
            pl.Time: "Time",
            pl.Array: "Array",
            pl.List: "List",
            pl.Struct: "Struct",
            pl.String: "String",
            pl.Categorical: "Categorical",
            pl.Enum: "Enum",
            pl.Utf8: "Utf8",
            pl.Binary: "Binary",
            pl.Boolean: "Boolean",
            pl.Null: "Null",
            pl.Object: "Object",
            pl.Unknown: "Unknown",
        }

        return dtype_map.get(dtype)

    @staticmethod
    def find_feat_occur(regex_path: str, ending_with: str) -> pl.DataFrame:
        """
        Finds occurrences of features ending with a specific string in Parquet files.

        Args:
        - regex_path (str): Regular expression to match Parquet file paths.
        - ending_with (str): Ending to filter feature names.
        Args:
        - regex_path (str): Regular expression to match Parquet file paths.
        - ending_with (str): Ending to filter feature names.

        Returns:
        - pl.DataFrame: DataFrame containing feature definitions, data types, and file locations.
        """
        feat_defs: pl.DataFrame = pl.read_csv(ROOT / "feature_definitions.csv").filter(
            pl.col("Variable").apply(lambda var: var.endswith(ending_with))
        )
        feat_defs.sort(by=["Variable"])

        feats: list[pl.String] = feat_defs["Variable"].to_list()
        feats.sort()

        occurrences: list[list] = [[set(), set()] for _ in range(feat_defs.height)]

        for path in glob(str(regex_path)):
            df_schema: dict = pl.read_parquet_schema(path)

            for feat, dtype in df_schema.items():
                index: int = Utility.find_index(feats, feat)
                if index != None:
                    occurrences[index][0].add(Utility.dtype_to_str(dtype))
                    occurrences[index][1].add(Path(path).stem)

        data_types: list[str] = [None] * feat_defs.height
        file_locs: list[str] = [None] * feat_defs.height

        for i, feat in enumerate(feats):
            data_types[i] = list(occurrences[i][0])
            file_locs[i] = list(occurrences[i][1])

        feat_defs = feat_defs.with_columns(pl.Series(data_types).alias("Data_Type(s)"))
        feat_defs = feat_defs.with_columns(pl.Series(file_locs).alias("File_Loc(s)"))

        #         feat_defs = feat_defs.with_columns(pl.Series(data_types).alias("Data_Type(s)"))
        #         feat_defs = feat_defs.with_columns(pl.Series(file_locs).alias("File_Loc(s)"))

        return feat_defs

    def reduce_memory_usage(df: pl.DataFrame, name) -> pl.DataFrame:
        """
        Returns:
        - pl.DataFrame: Optimized DataFrame.
        """
        print(
            f"Memory usage of dataframe \"{name}\" is {round(df.estimated_size('mb'), 4)} MB."
        )
        int_types = [
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        ]
        float_types = [pl.Float32, pl.Float64]

        for col in df.columns:
            col_type = df[col].dtype
            if col_type in int_types + float_types:
                c_min = df[col].min()
                c_max = df[col].max()

                if c_min is not None and c_max is not None:
                    if col_type in int_types:
                        if c_min >= 0:
                            if (
                                    c_min >= np.iinfo(np.uint8).min
                                    and c_max <= np.iinfo(np.uint8).max
                            ):
                                df = df.with_columns(df[col].cast(pl.UInt8))
                            elif (
                                    c_min >= np.iinfo(np.uint16).min
                                    and c_max <= np.iinfo(np.uint16).max
                            ):
                                df = df.with_columns(df[col].cast(pl.UInt16))
                            elif (
                                    c_min >= np.iinfo(np.uint32).min
                                    and c_max <= np.iinfo(np.uint32).max
                            ):
                                df = df.with_columns(df[col].cast(pl.UInt32))
                            elif (
                                    c_min >= np.iinfo(np.uint64).min
                                    and c_max <= np.iinfo(np.uint64).max
                            ):
                                df = df.with_columns(df[col].cast(pl.UInt64))
                        else:
                            if (
                                    c_min >= np.iinfo(np.int8).min
                                    and c_max <= np.iinfo(np.int8).max
                            ):
                                df = df.with_columns(df[col].cast(pl.Int8))
                            elif (
                                    c_min >= np.iinfo(np.int16).min
                                    and c_max <= np.iinfo(np.int16).max
                            ):
                                df = df.with_columns(df[col].cast(pl.Int16))
                            elif (
                                    c_min >= np.iinfo(np.int32).min
                                    and c_max <= np.iinfo(np.int32).max
                            ):
                                df = df.with_columns(df[col].cast(pl.Int32))
                            elif (
                                    c_min >= np.iinfo(np.int64).min
                                    and c_max <= np.iinfo(np.int64).max
                            ):
                                df = df.with_columns(df[col].cast(pl.Int64))
                    elif col_type in float_types:
                        if (
                                c_min > np.finfo(np.float32).min
                                and c_max < np.finfo(np.float32).max
                        ):
                            df = df.with_columns(df[col].cast(pl.Float32))

        print(
            f"Memory usage of dataframe \"{name}\" became {round(df.estimated_size('mb'), 4)} MB."
        )

        return df

    def to_pandas(df: pl.DataFrame, cat_cols: list[str] = None) -> (pd.DataFrame, list[str]):  # type: ignore
        """
        Converts a Polars DataFrame to a Pandas DataFrame.

        Args:): List of categorical columns. Default is None.

        Returns:
        - (pd.DataFrame, list[str]): Tuple containing the converted Pandas DataFrame and categorical columns.
        """
        df: pd.DataFrame = df.to_pandas()

        if cat_cols is None:
            cat_cols = list(df.select_dtypes("object").columns)

        df[cat_cols] = df[cat_cols].astype("str")

        return df, cat_cols


class Aggregator:
    @staticmethod
    def max_expr(df: pl.LazyFrame) -> list[pl.Series]:
        """
        Generates expressions for calculating maximum values for specific columns.

        Args:
        - df (pl.LazyFrame): Input LazyFrame.

        Returns:
        - list[pl.Series]: List of expressions for maximum values.
        """
        cols: list[str] = [
            col
            for col in df.columns
            if (col[-1] in ("P", "M", "A", "D", "T", "L")) or ("num_group" in col)
        ]

        expr_max: list[pl.Series] = [
            pl.col(col).max().alias(f"max_{col}") for col in cols
        ]
        return expr_max

    @staticmethod
    def min_expr(df: pl.LazyFrame) -> list[pl.Series]:
        """
        Generates expressions for calculating minimum values for specific columns.

        Args:
        - df (pl.LazyFrame): Input LazyFrame.

        Returns:
        - list[pl.Series]: List of expressions for minimum values.
        """
        cols: list[str] = [
            col
            for col in df.columns
            if (col[-1] in ("P", "M", "A", "D", "T", "L")) or ("num_group" in col)
        ]

        expr_min: list[pl.Series] = [
            pl.col(col).min().alias(f"min_{col}") for col in cols
        ]

        return expr_min

    @staticmethod
    def var_expr(df: pl.LazyFrame) -> list[pl.Series]:
        """
        Generates expressions for calculating variance for specific columns.

        Args:
        - df (pl.LazyFrame): Input LazyFrame.

        Returns:
        - list[pl.Series]: List of expressions for variance.
        """
        cols: list[str] = [col for col in df.columns if col.endswith(("P", "A", "D"))]

        expr_mean: list[pl.Series] = [
            pl.col(col).var().alias(f"var_{col}") for col in cols
        ]

        return expr_mean

    @staticmethod
    def mean_expr(df: pl.LazyFrame) -> list[pl.Series]:
        """
        Generates expressions for calculating mean values for specific columns.

        Args:
        - df (pl.LazyFrame): Input LazyFrame.

        Returns:
        - list[pl.Series]: List of expressions for mean values.
        """
        cols: list[str] = [col for col in df.columns if col.endswith(("P", "A", "D"))]

        expr_mean: list[pl.Series] = [
            pl.col(col).mean().alias(f"mean_{col}") for col in cols
        ]

        return expr_mean

    #     @staticmethod
    #     def mean_expr(df: pl.LazyFrame) -> list[pl.Series]:
    #         """
    #         Generates expressions for calculating mean values for specific columns.

    #         Args:
    #         - df (pl.LazyFrame): Input LazyFrame.

    #         Returns:
    #         - list[pl.Series]: List of expressions for mean values.
    #         """
    #         cols: list[str] = [col for col in df.columns if col.endswith(("P", "A", "D"))]

    #         expr_mean: list[pl.Series] = [
    #             pl.col(col).mean().alias(f"mean_{col}") for col in cols
    #         ]

    #         return expr_mean

    @staticmethod
    def mode_expr(df: pl.LazyFrame) -> list[pl.Series]:
        """
        Generates expressions for calculating mode values for specific columns.

        Args:
        - df (pl.LazyFrame): Input LazyFrame.

        Returns:
        - list[pl.Series]: List of expressions for mode values.
        """
        cols: list[str] = [col for col in df.columns if col.endswith("M")]

        expr_mode: list[pl.Series] = [
            pl.col(col).drop_nulls().mode().first().alias(f"mode_{col}") for col in cols
        ]

        return expr_mode

    @staticmethod
    def get_exprs(df: pl.LazyFrame) -> list[pl.Series]:
        """
        Combines expressions for maximum, mean, and variance calculations.

        Args:
        - df (pl.LazyFrame): Input LazyFrame.

        Returns:
        - list[pl.Series]: List of combined expressions.
        """
        exprs = (
                Aggregator.max_expr(df) + Aggregator.mean_expr(df) + Aggregator.var_expr(df)
        )

        return exprs


class SchemaGen:
    @staticmethod
    def change_dtypes(df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Changes the data types of columns in the DataFrame.

        Args:
        - df (pl.LazyFrame): Input LazyFrame.

        Returns:
        - pl.LazyFrame: LazyFrame with modified data types.
        """
        for col in df.columns:
            if col == "case_id":
                df = df.with_columns(pl.col(col).cast(pl.UInt32).alias(col))
            elif col in ["WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.UInt16).alias(col))
            elif col == "date_decision" or col[-1] == "D":
                df = df.with_columns(pl.col(col).cast(pl.Date).alias(col))
            elif col[-1] in ["P", "A"]:
                df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
        return df

    @staticmethod
    def scan_files(glob_path: str, depth: int = None) -> pl.LazyFrame:
        """
        Scans Parquet files matching the glob pattern and combines them into a LazyFrame.

        Args:
        - glob_path (str): Glob pattern to match Parquet files.
        - depth (int, optional): Depth level for data aggregation. Defaults to None.

        Returns:
        - pl.LazyFrame: Combined LazyFrame.
        """
        chunks: list[pl.LazyFrame] = []
        for path in glob(str(glob_path)):
            df: pl.LazyFrame = pl.scan_parquet(
                path, low_memory=True, rechunk=True
            ).pipe(SchemaGen.change_dtypes)
            print(f"File {Path(path).stem} loaded into memory.")

            if depth in (1, 2):
                exprs: list[pl.Series] = Aggregator.get_exprs(df)
                df = df.group_by("case_id").agg(exprs)

                del exprs
                gc.collect()

            chunks.append(df)

        df = pl.concat(chunks, how="vertical_relaxed")

        del chunks
        gc.collect()

        df = df.unique(subset=["case_id"])

        return df

    @staticmethod
    def join_dataframes(
            df_base: pl.LazyFrame,
            depth_0: list[pl.LazyFrame],
            depth_1: list[pl.LazyFrame],
            depth_2: list[pl.LazyFrame],
    ) -> pl.DataFrame:
        """
        Joins multiple LazyFrames with a base LazyFrame.

        Args:
        - df_base (pl.LazyFrame): Base LazyFrame.
        - depth_0 (list[pl.LazyFrame]): List of LazyFrames for depth 0.
        - depth_1 (list[pl.LazyFrame]): List of LazyFrames for depth 1.
        - depth_2 (list[pl.LazyFrame]): List of LazyFrames for depth 2.

        Returns:
        - pl.DataFrame: Joined DataFrame.
        """
        for i, df in enumerate(depth_0 + depth_1 + depth_2):
            df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")

        return df_base.collect().pipe(Utility.reduce_memory_usage, "df_train")


def filter_cols(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filters columns in the DataFrame based on null percentage and unique values for string columns.

    Args:
    Returns:
    - pl.DataFrame: DataFrame with filtered columns.
    """
    for col in df.columns:
        if col not in ["case_id", "year", "month", "week_num", "target"]:
            null_pct = df[col].is_null().mean()

            if null_pct > 0.95:
                df = df.drop(col)

    for col in df.columns:
        if (col not in ["case_id", "year", "month", "week_num", "target"]) & (
                df[col].dtype == pl.String
        ):
            freq = df[col].n_unique()

            if (freq > 200) | (freq == 1):
                df = df.drop(col)

    return df


def transform_cols(df: pl.DataFrame) -> pl.DataFrame:
    """
    Transforms columns in the DataFrame according to predefined rules.

    Args:
    Returns:
    - pl.DataFrame: DataFrame with transformed columns.
    """
    if "riskassesment_302T" in df.columns:
        if df["riskassesment_302T"].dtype == pl.Null:
            df = df.with_columns(
                [
                    pl.Series(
                        "riskassesment_302T_rng", df["riskassesment_302T"], pl.UInt8
                    ),
                    pl.Series(
                        "riskassesment_302T_mean", df["riskassesment_302T"], pl.UInt8
                    ),
                ]
            )
        else:
            pct_low: pl.Series = (
                df["riskassesment_302T"]
                .str.split(" - ")
                .apply(lambda x: x[0].replace("%", ""))
                .cast(pl.UInt8)
            )
            pct_high: pl.Series = (
                df["riskassesment_302T"]
                .str.split(" - ")
                .apply(lambda x: x[1].replace("%", ""))
                .cast(pl.UInt8)
            )

            diff: pl.Series = pct_high - pct_low
            avg: pl.Series = ((pct_low + pct_high) / 2).cast(pl.Float32)

            del pct_high, pct_low
            gc.collect()

            df = df.with_columns(
                [
                    diff.alias("riskassesment_302T_rng"),
                    avg.alias("riskassesment_302T_mean"),
                ]
            )

        df.drop("riskassesment_302T")

    return df


def handle_dates(df: pl.DataFrame) -> pl.DataFrame:
    """
    Handles date columns in the DataFrame.
    olumns.
    """
    for col in df.columns:
        if col.endswith("D"):
            df = df.with_columns(pl.col(col) - pl.col("date_decision"))
            df = df.with_columns(pl.col(col).dt.total_days().cast(pl.Int32))

    df = df.rename(
        {
            "MONTH": "month",
            "WEEK_NUM": "week_num"
        }
    )

    df = df.with_columns(
        [
            pl.col("date_decision").dt.year().alias("year").cast(pl.Int16),
            pl.col("date_decision").dt.day().alias("day").cast(pl.UInt8),
        ]
    )

    return df.drop("date_decision")


data_store: dict = {
    "df_base": SchemaGen.scan_files(TEST_DIR / "test_base.parquet"),
    "depth_0": [
        SchemaGen.scan_files(TEST_DIR / "test_static_cb_0.parquet"),
        SchemaGen.scan_files(TEST_DIR / "test_static_0_*.parquet"),
    ],
    "depth_1": [
        SchemaGen.scan_files(TEST_DIR / "test_applprev_1_*.parquet", 1),
        SchemaGen.scan_files(TEST_DIR / "test_tax_registry_a_1.parquet", 1),
        SchemaGen.scan_files(TEST_DIR / "test_tax_registry_b_1.parquet", 1),
        SchemaGen.scan_files(TEST_DIR / "test_tax_registry_c_1.parquet", 1),
        SchemaGen.scan_files(TEST_DIR / "test_credit_bureau_a_1_*.parquet", 1),
        SchemaGen.scan_files(TEST_DIR / "test_credit_bureau_b_1.parquet", 1),
        SchemaGen.scan_files(TEST_DIR / "test_other_1.parquet", 1),
        SchemaGen.scan_files(TEST_DIR / "test_person_1.parquet", 1),
        SchemaGen.scan_files(TEST_DIR / "test_deposit_1.parquet", 1),
        SchemaGen.scan_files(TEST_DIR / "test_debitcard_1.parquet", 1),
    ],
    "depth_2": [
        SchemaGen.scan_files(TEST_DIR / "test_credit_bureau_a_2_*.parquet", 2),
        SchemaGen.scan_files(TEST_DIR / "test_credit_bureau_b_2.parquet", 2),
    ],
}

df_test: pl.DataFrame = (
    SchemaGen.join_dataframes(**data_store)
    .pipe(transform_cols)
    .pipe(handle_dates)
    .pipe(Utility.reduce_memory_usage, "df_test")
)
del data_store
gc.collect()

print(f"Test data shape: {df_test.shape}")

filename = '/kaggle/input/newmodel1'

with open(f'{filename}/newmodel.pkl', 'rb') as f:
    [cat_cols, fitted_models_cat, fitted_models_lgb] = pickle.load(f)

df_test, cat_cols = Utility.to_pandas(df_test, cat_cols)


class VotingModel(BaseEstimator):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

    def predict_proba(self, X):
        numlen = len(fitted_models_cat)
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators[:numlen]]
        X[cat_cols] = X[cat_cols].astype("category")
        y_preds += [estimator.predict_proba(X) for estimator in self.estimators[numlen:]]

        return np.mean(y_preds, axis=0)


df_subm: pd.DataFrame = pd.read_csv(ROOT / "sample_submission.csv")
df_subm = df_subm.set_index("case_id")
model = VotingModel(fitted_models_cat + fitted_models_lgb)
X_test: pd.DataFrame = df_test.drop(columns=["week_num"]).set_index("case_id")
X_test = X_test[fitted_models_lgb[0].feature_name_]
y_pred = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
model = VotingModel(fitted_models_cat + fitted_models_lgb)
X_test: pd.DataFrame = df_test.drop(columns=["week_num"]).set_index("case_id")
X_test = X_test[fitted_models_lgb[0].feature_name_]
y_pred = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)

dffff = pd.read_csv("submission_codev2.csv").set_index("case_id")
df_subm["score"] = y_pred * 0.5 + dffff.score * 0.5
col = X_test.columns.tolist()[376]
condition = X_test[col] < (X_test[col].max() - X_test[col].min()) / 2 + X_test[col].min()
df_subm.loc[condition, 'score'] = (df_subm.loc[condition, 'score'] - 0.025).clip(0)
df_subm.to_csv("submission.csv")
