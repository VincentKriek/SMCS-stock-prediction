from pathlib import Path
import gc

import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Configuration
NEWS_N_ROWS = None

ROLLING_START_DATE = "2018-01-01"
ROLLING_END_DATE = "2023-12-31"
TRAIN_MONTHS = 6
VAL_MONTHS = 1
TEST_MONTHS = 5
MAX_SPLITS = None

BATCH_SIZE = 64
HIDDEN_DIM = 128
DROPOUT = 0.1
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5

TARGET_COL = "target_return"
PARQUET_PATH = "prepared_data_2018-01-01_2023-12-31.parquet"
OUTPUT_DIR = Path("output")

BASE_NUMERIC_FEATURES = ["open", "high", "low", "close", "adj close", "volume"]
LLM_SENTIMENT_MODE = "mean"  # "mean", "median", "mode", or None
BASELINE_MODELS = ["linear_regression", "mlp"]


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
        .filter(pl.col(TARGET_COL).is_not_null() & pl.col(TARGET_COL).is_finite())
    )


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
            if value != value:
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


# Split logic
def make_halfyear_rolling_splits(
    data: pl.LazyFrame,
    feature_cols: list[str],
    start_date: str,
    end_date: str,
    train_months: int,
    val_months: int,
    test_months: int,
):
    schema_names = data.collect_schema().names()
    needed_cols = ["Date", "Stock_symbol", TARGET_COL] + feature_cols
    used_cols = [c for c in needed_cols if c in schema_names]

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    df = (
        data.select(used_cols)
        .filter((pl.col("Date") >= pl.lit(start_dt)) & (pl.col("Date") <= pl.lit(end_dt)))
        .sort("Date")
        .collect(engine="streaming")
        .to_pandas()
    )

    if len(df) == 0:
        return []

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Stock_symbol"]).reset_index(drop=True)

    splits = []
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

        train_df = df[(df["Date"] >= train_start) & (df["Date"] <= train_end)].copy()
        val_df = df[(df["Date"] >= val_start) & (df["Date"] <= val_end)].copy()
        test_df = df[(df["Date"] >= test_start) & (df["Date"] <= test_end)].copy()

        if len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0:
            splits.append({"train_df": train_df, "val_df": val_df, "test_df": test_df})

        anchor = anchor + pd.DateOffset(months=test_months)
        if anchor > end_dt:
            break

    return splits


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
    save_path = OUTPUT_DIR / f"best_{model_name}_split_{split_idx}.pt"

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


# Main
def main():
    print("Running baseline models: linear regression + MLP")
    print(f"Parquet path: {PARQUET_PATH}")

    lf = pl.scan_parquet(PARQUET_PATH, n_rows=NEWS_N_ROWS)
    lf = add_next_day_return_target(lf)

    schema_names = lf.collect_schema().names()
    feature_cols = [c for c in BASE_NUMERIC_FEATURES if c in schema_names]

    if LLM_SENTIMENT_MODE == "mean" and "Sentiment_llm_mean_filled" in schema_names:
        feature_cols.append("Sentiment_llm_mean_filled")
    elif LLM_SENTIMENT_MODE == "median" and "Sentiment_llm_median_filled" in schema_names:
        feature_cols.append("Sentiment_llm_median_filled")
    elif LLM_SENTIMENT_MODE == "mode" and "Sentiment_llm_mode_filled" in schema_names:
        feature_cols.append("Sentiment_llm_mode_filled")

    if len(feature_cols) == 0:
        raise ValueError("No usable numeric feature columns were found in the parquet file.")

    print(f"Using features: {feature_cols}")

    rolling_splits = make_halfyear_rolling_splits(
        data=lf,
        feature_cols=feature_cols,
        start_date=ROLLING_START_DATE,
        end_date=ROLLING_END_DATE,
        train_months=TRAIN_MONTHS,
        val_months=VAL_MONTHS,
        test_months=TEST_MONTHS,
    )

    if len(rolling_splits) == 0:
        raise ValueError("No valid rolling splits were created.")

    if MAX_SPLITS is not None:
        rolling_splits = rolling_splits[:MAX_SPLITS]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Total splits to run: {len(rolling_splits)}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")

    summary_rows = []

    for split_idx, split_info in enumerate(rolling_splits, start=1):
        print(f"\nStarting split {split_idx}...")

        train_df = split_info["train_df"].copy()
        val_df = split_info["val_df"].copy()
        test_df = split_info["test_df"].copy()

        scaler = StandardScalerTorch()
        scaler.fit(train_df, feature_cols)
        train_df = scaler.transform(train_df, feature_cols)
        val_df = scaler.transform(val_df, feature_cols)
        test_df = scaler.transform(test_df, feature_cols)

        train_loader, val_loader, test_loader = build_loaders(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=feature_cols,
            batch_size=BATCH_SIZE,
        )

        if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0 or len(test_loader.dataset) == 0:
            print(f"Split {split_idx} is empty. Skipping.")
            continue

        test_rows = test_loader.dataset.rows

        for model_name in BASELINE_MODELS:
            model, test_loss, preds_all, targets_all = train_one_model(
                model_name=model_name,
                split_idx=split_idx,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
            )

            summary_rows.append({
                "model": model_name,
                "split": split_idx,
                "test_loss": test_loss,
            })

            results_df = pd.DataFrame(
                {
                    "Date": [row.get("Date") for row in test_rows],
                    "Stock_symbol": [row.get("Stock_symbol") for row in test_rows],
                    "target_return": targets_all,
                    "prediction": preds_all,
                }
            )
            csv_path = OUTPUT_DIR / f"preds_{model_name}_split_{split_idx}.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"Predictions saved to: {csv_path}")

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = OUTPUT_DIR / "baseline_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
