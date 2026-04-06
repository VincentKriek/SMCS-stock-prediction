from pathlib import Path
import gc

import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Configuration
NEWS_N_ROWS = None

ROLLING_START_DATE = "2018-01-01"
ROLLING_END_DATE = "2023-12-31"
TRAIN_MONTHS = 6
VAL_MONTHS = 1
TEST_MONTHS = 5
MAX_SPLITS = 1

BATCH_SIZE = 64
HIDDEN_DIM = 128
DROPOUT = 0.1
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5

TARGET_COL = "target_return"
STOCK_SYMBOL = "Stock_symbol"
PARQUET_PATH = "data/pre-processor/prepared_data_2018-01-01_2023-12-31.parquet"
TRAIN_PER_STOCK = False

BASE_NUMERIC_FEATURES = ["open", "high", "low", "close", "adj close", "volume"]

LLM_SENTIMENT_MODE = None  # "mean", "median", "mode", or None
BASELINE_MODELS = ["linear_regression", "mlp"]

output_dir = Path("data/model/output")

# Data utilities
def add_next_day_return_target(lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        lf.sort(["Stock_symbol", "Date"])
        .with_columns(
            (
                (pl.col("close").shift(-1).over("Stock_symbol") - pl.col("close"))
                / pl.col("close")
            ).alias(TARGET_COL)
        )
        .filter(
            pl.col(TARGET_COL).is_not_null() & 
            pl.col(TARGET_COL).is_finite() & 
            (pl.col(TARGET_COL).abs() <= 0.5)  # Filter outliers > 50% daily move
        )
    )


def scale_features_per_stock(lf: pl.LazyFrame, features: list[str]) -> pl.LazyFrame:
    """Z-score features relative to each stock's own mean/std."""
    return lf.with_columns([
        ((pl.col(f) - pl.col(f).mean().over("Stock_symbol")) / 
         (pl.col(f).std().over("Stock_symbol") + 1e-8)).alias(f)
        for f in features
    ])

# Split planning only, do not load all rows into pandas at once
def make_halfyear_rolling_split_plans(
    start_date: str,
    end_date: str,
    train_months: int,
    val_months: int,
    test_months: int,
):
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    split_plans = []
    anchor = start_dt

    while True:
        train_start = anchor
        train_end = train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + pd.DateOffset(months=val_months) - pd.Timedelta(days=1)
        test_start = val_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

        if val_start > end_dt or test_start > end_dt:
            break

        test_end = min(test_end, end_dt)

        split_plans.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )

        anchor = anchor + pd.DateOffset(months=test_months)
        if anchor > end_dt:
            break

    return split_plans


def collect_split_dataframe(
    lf: pl.LazyFrame,
    feature_cols: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    schema_names = lf.collect_schema().names()
    needed_cols = ["Date", STOCK_SYMBOL, TARGET_COL] + feature_cols
    used_cols = [c for c in needed_cols if c in schema_names]

    df = (
        lf.select(used_cols)
        .filter((pl.col("Date") >= pl.lit(start_date)) & (pl.col("Date") <= pl.lit(end_date)))
        .sort(["Date", "Stock_symbol"])
        .collect(engine="streaming")
        .to_pandas()
    )

    if len(df) == 0:
        return df

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Stock_symbol"]).reset_index(drop=True)
    return df

def train_baseline(lf: pl.LazyFrame, split_plans, feature_cols, use_lin_regr, stock_name=None):
    if stock_name:
        lf = lf.filter(pl.col(STOCK_SYMBOL).eq(stock_name))

    mse_per_split, r2_per_split = [], []
    for split_idx, plan in enumerate(split_plans, start=1):
        # print(f"\nStarting split {split_idx}...")
        # print(
        #     f"Train: {plan['train_start'].date()} -> {plan['train_end'].date()} | "
        #     f"Val: {plan['val_start'].date()} -> {plan['val_end'].date()} | "
        #     f"Test: {plan['test_start'].date()} -> {plan['test_end'].date()}"
        # )

        train_df = collect_split_dataframe(
            lf=lf,
            feature_cols=feature_cols,
            start_date=plan["train_start"],
            end_date=plan["train_end"],
        )
        val_df = collect_split_dataframe(
            lf=lf,
            feature_cols=feature_cols,
            start_date=plan["val_start"],
            end_date=plan["val_end"],
        )
        test_df = collect_split_dataframe(
            lf=lf,
            feature_cols=feature_cols,
            start_date=plan["test_start"],
            end_date=plan["test_end"],
        )


        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            # print(f"Split {split_idx} is empty. Skipping.")
            del train_df, val_df, test_df
            gc.collect()
            continue

        # Concat train and val (sklearn handles internally with MLP)
        train_df = pd.concat([train_df, val_df], axis=0) # train = train + val

        drop_cols = ["target_return", "Date"]
        if stock_name:
            drop_cols.append(STOCK_SYMBOL)

        X_train, y_train = train_df.drop(columns=drop_cols), train_df["target_return"]
        X_test, y_test = test_df.drop(columns=drop_cols), test_df["target_return"]

        if not stock_name:
            # One-hot enocde stock-symbol
            preprocess = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore"), ["Stock_symbol"]), # use 0 for unseen stocks
                ],
                remainder="passthrough"
            )

        if use_lin_regr:
            # print("===== Linear Regression =====")

            if not stock_name:
                lin_regr_model = Pipeline([
                    ("preprocess", preprocess),
                    ("regressor", LinearRegression())
                ])
            else:
                lin_regr_model = Pipeline([
                    ("regressor", LinearRegression())
                ])

            lin_regr_model.fit(X_train, y_train)

            preds = lin_regr_model.predict(X_test)

            # dates = test_df["Date"]
            # stocks = test_df["Stock_symbol"]
            # results_df = pd.DataFrame({
            #     "Date": dates,
            #     "Stock_symbol": stocks,
            #     "target_return": y_test,
            #     "prediction": preds
            # })
            # results_df = results_df.sort_values(by=["Stock_symbol", "Date"]).reset_index(drop=True)
            # csv_path = output_dir / f"BASE_preds_linregr_split_{split_idx}.csv"
            # results_df.to_csv(csv_path, index=False)
            # print(f"Predictions saved to: {csv_path}")

            mse = mean_squared_error(y_true=y_test, y_pred=preds)
            r2  = r2_score(y_true=y_test, y_pred=preds)
            mse_per_split.append(mse)
            r2_per_split.append(r2)
        else:
            # print("===== MLP =====")
            val_fraction = VAL_MONTHS / (TRAIN_MONTHS + VAL_MONTHS)
            if not stock_name:
                mlp_model = Pipeline([
                    ("preprocess", preprocess),
                    ("mlp", MLPRegressor(
                        hidden_layer_sizes=(32,),
                        max_iter=100,
                        validation_fraction=val_fraction,
                        early_stopping=True
                    ))
                ])
            else:
                mlp_model = Pipeline([
                    ("mlp", MLPRegressor(
                        hidden_layer_sizes=(32,),
                        max_iter=100,
                        validation_fraction=val_fraction,
                        early_stopping=True
                    ))
                ])

            mlp_model.fit(X_train, y_train)

            preds = mlp_model.predict(X_test)

            # dates = test_df["Date"]
            # stocks = test_df["Stock_symbol"]
            # results_df = pd.DataFrame({
            #     "Date": dates,
            #     "Stock_symbol": stocks,
            #     "target_return": y_test,
            #     "prediction": preds
            # })
            # results_df = results_df.sort_values(by=["Stock_symbol", "Date"]).reset_index(drop=True)

            # csv_path = output_dir / f"BASE_preds_mlp_split_{split_idx}.csv"
            # results_df.to_csv(csv_path, index=False)
            # print(f"Predictions saved to: {csv_path}")

            mse = mean_squared_error(y_true=y_test, y_pred=preds)
            r2  = r2_score(y_true=y_test, y_pred=preds)
            mse_per_split.append(mse)
            r2_per_split.append(r2)

    return mse_per_split, r2_per_split


# region Main
# Main
def main():
    print("Running baseline models: linear regression + MLP")
    print(f"Parquet path: {PARQUET_PATH}")

    lf = pl.scan_parquet(PARQUET_PATH, n_rows=NEWS_N_ROWS)
    lf = add_next_day_return_target(lf)

    schema_names = lf.collect_schema().names()
    feature_cols = [c for c in BASE_NUMERIC_FEATURES if c in schema_names]

    if len(feature_cols) == 0:
        raise ValueError("No usable numeric feature columns were found in the parquet file.")

    # APPLY PER-STOCK SCALING (CRITICAL)
    print(f"Applying per-stock scaling to: {feature_cols}")
    lf = scale_features_per_stock(lf, feature_cols)

    print(f"Using features: {feature_cols}")

    split_plans = make_halfyear_rolling_split_plans(
        start_date=ROLLING_START_DATE,
        end_date=ROLLING_END_DATE,
        train_months=TRAIN_MONTHS,
        val_months=VAL_MONTHS,
        test_months=TEST_MONTHS,
    )

    if len(split_plans) == 0:
        raise ValueError("No valid rolling split plans were created.")

    if MAX_SPLITS is not None:
        split_plans = split_plans[:MAX_SPLITS]

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Total splits to run: {len(split_plans)}")
    print(f"Output folder: {output_dir.resolve()}")

    TRAIN_PER_STOCK = False
    USE_LIN_REGR = False
    if TRAIN_PER_STOCK:
        stocks = lf.select(STOCK_SYMBOL).unique().collect()[STOCK_SYMBOL].to_list()
        res = {s: ([], []) for s in stocks}
        for s in stocks[:10]:
            s_mses, s_r2s = train_baseline(lf, split_plans, feature_cols,use_lin_regr=USE_LIN_REGR, stock_name=s)
            res[s] = (s_mses, s_r2s)

        # Write results
        rows = []
        for s, (mses, r2s) in res.items():
            for i, (mse, r2) in enumerate(zip(mses, r2s)):
                rows.append({
                    "split_idx": i,
                    "mse": mse,
                    "r2": r2
                })
        df = pd.DataFrame(rows)
        model_name = "LINREGR" if USE_LIN_REGR else "MLP"
        df.to_csv(output_dir / f"BASE_per_stock_{model_name}_metrics.csv", index=False)
    else:
        mses, r2s = train_baseline(lf, split_plans, feature_cols, use_lin_regr=USE_LIN_REGR, stock_name=None)

        # Write results
        rows = []
        for i, (mse, r2) in enumerate(zip(mses, r2s)):
            rows.append({
                "split_idx": i,
                "mse": mse,
                "r2": r2
            })
        df = pd.DataFrame(rows)
        model_name = "LINREGR" if USE_LIN_REGR else "MLP"
        df.to_csv(output_dir / f"BASE_global_{model_name}_metrics.csv", index=False)


if __name__ == "__main__":
    main()


    
