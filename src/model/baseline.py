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
PARQUET_PATH = "data/pre-processor/prepared_data_2018-01-01_2023-12-31.parquet"

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


class TabularStockDataset(Dataset):
    def __init__(self, rows, feature_cols, target_col):
        self.rows = []
        for row in rows:
            target = row.get(target_col)
            if target is None:
                continue
            self.rows.append(row)

        self.feature_cols = feature_cols
        self.target_col = target_col

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        x = []
        for col in self.feature_cols:
            value = row.get(col, 0.0)
            try:
                value = float(value)
            except Exception:
                value = 0.0
            if value != value:  # NaN
                value = 0.0
            x.append(value)

        y = float(row[self.target_col])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class LinearRegressionBaseline(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class PureMLPBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class StandardScalerTorch:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, df: pd.DataFrame, feature_cols: list[str]):
        mean = df[feature_cols].mean(axis=0)
        std = df[feature_cols].std(axis=0).replace(0, 1.0).fillna(1.0)
        self.mean = mean
        self.std = std

    def transform(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        out = df.copy()
        out[feature_cols] = (out[feature_cols] - self.mean) / self.std
        out[feature_cols] = out[feature_cols].fillna(0.0)
        return out


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
    needed_cols = ["Date", "Stock_symbol", TARGET_COL] + feature_cols
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


# Training / evaluation
def build_loaders(train_df, val_df, test_df, feature_cols, batch_size):
    train_dataset = TabularStockDataset(train_df.to_dict("records"), feature_cols, TARGET_COL)
    val_dataset = TabularStockDataset(val_df.to_dict("records"), feature_cols, TARGET_COL)
    test_dataset = TabularStockDataset(test_df.to_dict("records"), feature_cols, TARGET_COL)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds_all = []
    targets_all = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        preds = model(x)
        loss = criterion(preds, y)
        total_loss += loss.item()
        preds_all.extend(preds.detach().cpu().tolist())
        targets_all.extend(y.detach().cpu().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, preds_all, targets_all


def build_model(model_name: str, input_dim: int):
    if model_name == "linear_regression":
        return LinearRegressionBaseline(input_dim=input_dim)
    if model_name == "mlp":
        return PureMLPBaseline(input_dim=input_dim, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    raise ValueError(f"Unsupported baseline model: {model_name}")


def train_one_model(model_name, split_idx, train_loader, val_loader, test_loader, device):
    model = build_model(model_name, input_dim=train_loader.dataset[0][0].shape[0]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    patience_counter = 0
    save_path = output_dir / f"best_{model_name}_split_{split_idx}.pt"

    epoch_bar = tqdm(range(MAX_EPOCHS), desc=f"{model_name} | Split {split_idx}", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        running_train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / max(len(train_loader), 1)
        avg_val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        epoch_bar.set_postfix(train=f"{avg_train_loss:.6f}", val=f"{avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"{model_name} early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, preds_all, targets_all = evaluate(model, test_loader, criterion, device)
    print(f"{model_name} | Split {split_idx} | Test={test_loss:.6f}")
    return model, test_loss, preds_all, targets_all

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

    for split_idx, plan in enumerate(split_plans, start=1):
        print(f"\nStarting split {split_idx}...")
        print(
            f"Train: {plan['train_start'].date()} -> {plan['train_end'].date()} | "
            f"Val: {plan['val_start'].date()} -> {plan['val_end'].date()} | "
            f"Test: {plan['test_start'].date()} -> {plan['test_end'].date()}"
        )

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
            print(f"Split {split_idx} is empty. Skipping.")
            del train_df, val_df, test_df
            gc.collect()
            continue

        # One-hot enocde stock-symbol
        preprocess = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["Stock_symbol"]), # use 0 for unseen stocks
                ("num", StandardScaler(), feature_cols)
            ],
            remainder="passthrough"
        )

        # Concat train and val (sklearn handles internally with MLP)
        train_df = pd.concat([train_df, val_df], axis=0) # train = train + val

        drop_cols = ["target_return", "Date"]
        X_train, y_train = train_df.drop(columns=drop_cols), train_df["target_return"]
        X_test, y_test = test_df.drop(columns=drop_cols), test_df["target_return"]

        # region Linear Regresson
        print("===== Linear Regression =====")

        lin_regr_model = Pipeline([
            ("preprocess", preprocess),
            ("regressor", LinearRegression())
        ])

        lin_regr_model.fit(X_train, y_train)

        coeff = lin_regr_model["regressor"].coef_
        print(f"No. of coeffs: {len(coeff)}")
        feature_names = lin_regr_model.named_steps["preprocess"].get_feature_names_out()
        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coeff": coeff
        })
        coef_df = coef_df.sort_values(by="coeff", key=abs, ascending=False)
        intercept_df = pd.DataFrame({
            "feature": ["intercept"],
            "coeff": [lin_regr_model.named_steps["regressor"].intercept_]
        })
        coef_df = pd.concat([intercept_df, coef_df])
        print(coef_df)

        preds = lin_regr_model.predict(X_test)

        dates = test_df["Date"]
        stocks = test_df["Stock_symbol"]
        results_df = pd.DataFrame({
            "Date": dates,
            "Stock_symbol": stocks,
            "target_return": y_test,
            "prediction": preds
        })
        results_df = results_df.sort_values(by=["Stock_symbol", "Date"]).reset_index(drop=True)

        csv_path = output_dir / f"BASE_preds_linregr_split_{split_idx}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Predictions saved to: {csv_path}")

        mse = mean_squared_error(y_true=y_test, y_pred=preds)
        r2  = r2_score(y_true=y_test, y_pred=preds)
        print("MSE: ", mse)
        print("R²:  ", r2)
        # endregion

        # region MLP
        print("===== MLP =====")
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

        dates = test_df["Date"]
        stocks = test_df["Stock_symbol"]
        results_df = pd.DataFrame({
            "Date": dates,
            "Stock_symbol": stocks,
            "target_return": y_test,
            "prediction": preds
        })
        results_df = results_df.sort_values(by=["Stock_symbol", "Date"]).reset_index(drop=True)

        csv_path = output_dir / f"BASE_preds_mlp_split_{split_idx}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Predictions saved to: {csv_path}")

        mse = mean_squared_error(y_true=y_test, y_pred=preds)
        r2  = r2_score(y_true=y_test, y_pred=preds)
        print("MSE: ", mse)
        print("R²:  ", r2)
        # endregion

if __name__ == "__main__":
    main()


    
