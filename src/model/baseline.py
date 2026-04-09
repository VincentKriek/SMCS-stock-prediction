from pathlib import Path
import gc

import pandas as pd
import polars as pl
import torch

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from sklearn.neural_network import MLPRegressor

# Configuration
NEWS_N_ROWS = None

ROLLING_START_DATE = "2018-01-01"
ROLLING_END_DATE = "2023-12-31"
TRAIN_MONTHS = 6
VAL_MONTHS = 1
TEST_MONTHS = 5
MAX_SPLITS = None

TARGET_COL = "target_return"
STOCK_SYMBOL = "Stock_symbol"
PARQUET_PATH = "data/pre-processor/prepared_data_2018-01-01_2023-12-31.parquet"
USE_LIN_REGR = False

BASE_NUMERIC_FEATURES = ["open", "high", "low", "close", "adj close", "volume"]
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

def train_baseline(lf: pl.LazyFrame, split_plans, feature_cols, use_lin_regr):
    mse_per_split, r2_per_split, mae_per_split = [], [], []
    for _, plan in enumerate(split_plans, start=1):
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
            del train_df, val_df, test_df
            gc.collect()
            continue

        # Concat train and val (sklearn handles internally with MLP)
        train_df = pd.concat([train_df, val_df], axis=0) # train = train + val

        drop_cols = ["target_return", "Date"]

        X_train, y_train = train_df.drop(columns=drop_cols), train_df["target_return"]
        X_test, y_test = test_df.drop(columns=drop_cols), test_df["target_return"]

        # One-hot enocde stock-symbol
        preprocess = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["Stock_symbol"]), # use 0 for unseen stocks
            ],
            remainder="passthrough"
        )

        if use_lin_regr: # Linear Regression
            lin_regr_model = Pipeline([
                ("preprocess", preprocess),
                ("regressor", LinearRegression())
            ])

            lin_regr_model.fit(X_train, y_train)

            preds = lin_regr_model.predict(X_test)

            mse = mean_squared_error(y_true=y_test, y_pred=preds)
            r2  = r2_score(y_true=y_test, y_pred=preds)
            mae = median_absolute_error(y_true=y_test, y_pred=preds)
            mse_per_split.append(mse)
            r2_per_split.append(r2)
            mae_per_split.append(mae)
        else: # MLP
            val_fraction = VAL_MONTHS / (TRAIN_MONTHS + VAL_MONTHS)
            mlp_model = Pipeline([
                ("preprocess", preprocess),
                ("mlp", MLPRegressor(
                    hidden_layer_sizes=(32,),
                    max_iter=100,
                    validation_fraction=val_fraction,
                    early_stopping=True
                ))
            ])

            mlp_model.fit(X_train, y_train)

            preds = mlp_model.predict(X_test)

            mse = mean_squared_error(y_true=y_test, y_pred=preds)
            r2  = r2_score(y_true=y_test, y_pred=preds)
            mae = median_absolute_error(y_true=y_test, y_pred=preds)
            mse_per_split.append(mse)
            r2_per_split.append(r2)
            mae_per_split.append(mae)

    return mse_per_split, r2_per_split, mae_per_split


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

    mses, r2s, maes = train_baseline(lf, split_plans, feature_cols, use_lin_regr=USE_LIN_REGR)

    # Write results
    rows = []
    for i, (mse, r2, mae) in enumerate(zip(mses, r2s, maes)):
        rows.append({
            "split_idx": i,
            "mse": mse,
            "medae": mae,
            "r2": r2
        })
    df = pd.DataFrame(rows)
    model_name = "LINREGR" if USE_LIN_REGR else "MLP"
    df.to_csv(output_dir / f"BASE_{model_name}_metrics.csv", index=False)


if __name__ == "__main__":
    main()


    
