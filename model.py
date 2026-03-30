from lstm import LazyHeadlineVectorizer, LSTM_Encoder
from Mdgnn import MDGNN
from graph_snapshot import build_quarter_snapshots, expand_quarter_snapshots_to_daily

import polars as pl
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import tqdm



# Configuration
HIDDEN_DIM = 128
GNN_LAYERS = 2
WINDOW_SIZE = 10
MAX_EPOCHS = 500
PATIENCE = 20
BATCH_SIZE = 8

FEATURE_COLS = ["open", "high"]
TARGET_COL = "close"

ROLLING_START_DATE = "2020-01-01"
ROLLING_END_DATE = "2023-02-28"

TRAIN_MONTHS = 6
VAL_MONTHS = 1
TEST_MONTHS = 6



# Graph cache
class DailyGraphFeatureCache:
    def __init__(
        self,
        mdgnn_model: MDGNN,
        nodes_bank_path: str,
        nodes_stock_path: str,
        edges_path: str,
        trading_dates,
        device: torch.device,
        hidden_dim: int = 128,
    ):
        self.device = device
        self.hidden_dim = hidden_dim
        self.cache = {}

        nodes_bank_df = pd.read_parquet(nodes_bank_path)
        nodes_stock_df = pd.read_parquet(nodes_stock_path)
        edges_df = pd.read_parquet(edges_path)

        quarter_snapshots = build_quarter_snapshots(
            nodes_bank_df,
            nodes_stock_df,
            edges_df
        )

        daily_snapshots = expand_quarter_snapshots_to_daily(
            trading_dates=trading_dates,
            quarter_snapshots=quarter_snapshots
        )

        mdgnn_model = mdgnn_model.to(device)
        mdgnn_model.eval()

        with torch.no_grad():
            for dt, snap in daily_snapshots.items():
                stock_feat = snap["stock_feat"].to(device)
                bank_feat = snap["bank_feat"].to(device) if snap["bank_feat"] is not None else None

                industry_feat = snap["industry_feat"]
                if industry_feat is not None:
                    industry_feat = industry_feat.to(device)

                edges = {}
                for rel, value in snap["edges"].items():
                    if value is None:
                        edges[rel] = None
                    else:
                        edge_index, edge_attr = value
                        edges[rel] = (
                            edge_index.to(device),
                            edge_attr.to(device) if edge_attr is not None else None
                        )

                stock_emb = mdgnn_model.encode_snapshot(
                    stock_feat=stock_feat,
                    bank_feat=bank_feat,
                    industry_feat=industry_feat,
                    edges=edges
                )  # [num_stocks, hidden_dim]

                pooled = stock_emb.mean(dim=0).detach().cpu()  # [hidden_dim]
                self.cache[pd.Timestamp(dt).normalize()] = pooled

        self.sorted_dates = sorted(self.cache.keys())

    def lookup(self, date_value):
        dt = pd.Timestamp(date_value).normalize()
        if dt not in self.cache:
            return torch.zeros(self.hidden_dim, dtype=torch.float32)
        return self.cache[dt].clone().float()

    def lookup_window(self, date_value, window_size=10):
        dt = pd.Timestamp(date_value).normalize()
        valid_dates = [d for d in self.sorted_dates if d <= dt]

        if len(valid_dates) == 0:
            return torch.zeros(window_size, self.hidden_dim, dtype=torch.float32)

        chosen = valid_dates[-window_size:]
        seq = [self.cache[d].clone().float() for d in chosen]

        if len(seq) < window_size:
            pad_len = window_size - len(seq)
            pads = [torch.zeros(self.hidden_dim, dtype=torch.float32) for _ in range(pad_len)]
            seq = pads + seq

        return torch.stack(seq, dim=0)  # [window_size, hidden_dim]



# Dataset
class NewsGraphDataset(Dataset):
    def __init__(
        self,
        rows,
        stock2id: dict[str, int],
        graph_cache: DailyGraphFeatureCache,
        embedding_col: str,
        feature_cols: list[str],
        target_col: str,
        max_headline_len: int,
        graph_hidden_dim: int = 128,
        window_size: int = 10
    ):
        self.samples = []
        self.window_size = window_size
        self.graph_hidden_dim = graph_hidden_dim

        for row in rows:
            target = row.get(target_col)
            if target is None:
                continue

            stock_symbol = row.get("Stock_symbol")
            if stock_symbol not in stock2id:
                continue

            text_ids = row.get(embedding_col)
            if text_ids is None:
                text_ids = [0] * max_headline_len
            else:
                if len(text_ids) < max_headline_len:
                    text_ids = text_ids + [0] * (max_headline_len - len(text_ids))
                else:
                    text_ids = text_ids[:max_headline_len]

            numeric_feats = []
            for col in feature_cols:
                val = row.get(col)
                if val is None:
                    val = -1.0
                numeric_feats.append(float(val))

            date_value = row.get("Date")
            graph_seq = graph_cache.lookup_window(date_value, window_size=window_size)

            self.samples.append({
                "text_ids": torch.tensor(text_ids, dtype=torch.long),
                "numeric_feats": torch.tensor(numeric_feats, dtype=torch.float32),
                "target": torch.tensor(float(target), dtype=torch.float32),
                "stock_id": torch.tensor(stock2id[stock_symbol], dtype=torch.long),
                "graph_seq": graph_seq if graph_seq is not None else torch.zeros(window_size, graph_hidden_dim, dtype=torch.float32),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s["text_ids"],
            s["numeric_feats"],
            s["target"],
            s["stock_id"],
            s["graph_seq"],
        )



# Fusion model
class LSTM_MDGNN_Fusion(nn.Module):
    def __init__(
        self,
        lstm_encoder: LSTM_Encoder,
        mdgnn_model: MDGNN,
        num_numeric_features: int,
        graph_hidden_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        task: str = "regression",
    ):
        super().__init__()
        assert task in ["regression", "classification"]

        self.lstm_encoder = lstm_encoder
        self.mdgnn_model = mdgnn_model
        self.task = task

        self.numeric_proj = nn.Sequential(
            nn.Linear(num_numeric_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.graph_proj = nn.Sequential(
            nn.Linear(graph_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, text_ids, numeric_feats, stock_ids, graph_seq):
        # Text branch
        text_repr = self.lstm_encoder(text_ids, stock_ids)   # [B, H]

        # Numeric branch
        num_repr = self.numeric_proj(numeric_feats)          # [B, H]

        # Graph temporal branch
        # graph_seq: [B, T, H]
        _, graph_repr, _ = self.mdgnn_model.forward_from_sequence(graph_seq, return_attention=True)
        graph_repr = self.graph_proj(graph_repr)             # [B, H]

        fused = torch.cat([text_repr, num_repr, graph_repr], dim=1)
        out = self.fusion_head(fused).squeeze(-1)

        if self.task == "classification":
            return torch.sigmoid(out)
        return out



# Rolling split helpers
def make_halfyear_rolling_splits(
    data: pl.LazyFrame,
    start_date: str = "2018-01-01",
    end_date: str = "2023-02-28",
    train_months: int = 6,
    val_months: int = 1,
    test_months: int = 6,
):
    df = data.sort("Date").collect(engine="streaming").to_pandas()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[
        (df["Date"] >= pd.Timestamp(start_date)) &
        (df["Date"] <= pd.Timestamp(end_date))
    ].copy()

    splits = []
    anchor = pd.Timestamp(start_date)

    while True:
        # Train: first 6 months
        train_start = anchor
        train_end = train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)

        # Validation: the next 1 month after training
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + pd.DateOffset(months=val_months) - pd.Timedelta(days=1)

        # Test: the following 6 months after validation
        test_start = val_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

        # Stop if validation or test period starts beyond available data
        if val_start > pd.Timestamp(end_date) or test_start > pd.Timestamp(end_date):
            break

        test_end = min(test_end, pd.Timestamp(end_date))

        train_df = df[(df["Date"] >= train_start) & (df["Date"] <= train_end)].copy()
        val_df = df[(df["Date"] >= val_start) & (df["Date"] <= val_end)].copy()
        test_df = df[(df["Date"] >= test_start) & (df["Date"] <= test_end)].copy()

        if len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0:
            splits.append({
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_rows": train_df.to_dict("records"),
                "val_rows": val_df.to_dict("records"),
                "test_rows": test_df.to_dict("records"),
            })

        # Move anchor forward by 6 months: one model every half year
        anchor = anchor + pd.DateOffset(months=test_months)

        if anchor > pd.Timestamp(end_date):
            break

    return splits


def build_loaders_for_split(
    split_dict,
    stock2id,
    graph_cache,
    batch_size,
    max_headline_len,
    feature_cols,
    target_col,
    window_size=10,
):
    train_dataset = NewsGraphDataset(
        rows=split_dict["train_rows"],
        stock2id=stock2id,
        graph_cache=graph_cache,
        embedding_col="embedded_headline",
        feature_cols=feature_cols,
        target_col=target_col,
        max_headline_len=max_headline_len,
        window_size=window_size,
    )

    val_dataset = NewsGraphDataset(
        rows=split_dict["val_rows"],
        stock2id=stock2id,
        graph_cache=graph_cache,
        embedding_col="embedded_headline",
        feature_cols=feature_cols,
        target_col=target_col,
        max_headline_len=max_headline_len,
        window_size=window_size,
    )

    test_dataset = NewsGraphDataset(
        rows=split_dict["test_rows"],
        stock2id=stock2id,
        graph_cache=graph_cache,
        embedding_col="embedded_headline",
        feature_cols=feature_cols,
        target_col=target_col,
        max_headline_len=max_headline_len,
        window_size=window_size,
    )

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )



# Main
if __name__ == "__main__":
    # 1. Load and vectorize prepared data
    l = LazyHeadlineVectorizer(
        "../prepared_data_2018-01-01_2023-12-31.parquet",
        n_rows=1000
    )
    l.run()

    # Optional cleanup if these columns exist
    drop_cols = [
        "Sentiment_llm_mean_filled",
        "Sentiment_llm_median_filled",
        "Sentiment_llm_mode_filled"
    ]
    schema_names = l.lf.collect_schema().names()
    existing_drop_cols = [c for c in drop_cols if c in schema_names]
    if existing_drop_cols:
        l.lf = l.lf.drop(*existing_drop_cols)


    # 2. Build stock symbol -> integer id for LSTM branch
    stocks = sorted(l.lf.select("Stock_symbol").unique().collect().to_series().to_list())
    stock2id = {symbol: idx for idx, symbol in enumerate(stocks)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # 3. Prepare one MDGNN model for graph cache building
    mdgnn_for_cache = MDGNN(
        stock_in_dim=4,
        bank_in_dim=4,
        industry_in_dim=1,
        edge_dims={"SB": 5, "BS": 5},
        hidden_dim=HIDDEN_DIM,
        gnn_layers=GNN_LAYERS,
        num_heads=4,
        ff_dim=256,
        dropout=0.1,
        task="regression"
    ).to(device)

    trading_dates = (
        l.lf.select("Date")
        .unique()
        .sort("Date")
        .collect()
        .to_series()
        .to_list()
    )

    graph_cache = DailyGraphFeatureCache(
        mdgnn_model=mdgnn_for_cache,
        nodes_bank_path="nodes_bank.parquet",
        nodes_stock_path="nodes_stock.parquet",
        edges_path="edges_bank_stock.parquet",
        trading_dates=trading_dates,
        device=device,
        hidden_dim=HIDDEN_DIM,
    )


    # 4. Create rolling splits
    rolling_splits = make_halfyear_rolling_splits(
        data=l.lf,
        start_date=ROLLING_START_DATE,
        end_date=ROLLING_END_DATE,
        train_months=TRAIN_MONTHS,
        val_months=VAL_MONTHS,
        test_months=TEST_MONTHS,
    )

    print(f"Number of rolling models: {len(rolling_splits)}")

    all_test_losses = []


    # 5. Train one model per 6-month rolling period
    for split_idx, split_info in enumerate(rolling_splits, start=1):
        print("\n" + "=" * 60)
        print(f"Rolling model {split_idx}")
        print(f"Train: {split_info['train_start'].date()} -> {split_info['train_end'].date()}")
        print(f"Val:   {split_info['val_start'].date()} -> {split_info['val_end'].date()}")
        print(f"Test:  {split_info['test_start'].date()} -> {split_info['test_end'].date()}")

        train_loader, val_loader, test_loader = build_loaders_for_split(
            split_dict=split_info,
            stock2id=stock2id,
            graph_cache=graph_cache,
            batch_size=BATCH_SIZE,
            max_headline_len=l.max_headline_len,
            feature_cols=FEATURE_COLS,
            target_col=TARGET_COL,
            window_size=WINDOW_SIZE,
        )

        if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0 or len(test_loader.dataset) == 0:
            print(f"Skipping split {split_idx} because one dataset is empty.")
            continue

        # Reinitialize model every 6 months
        lstm_encoder = LSTM_Encoder(
            vocab_size=len(l.word2id),
            embedding_dim=l.vector_size,
            hidden_dim=HIDDEN_DIM,
            embedding_matrix=l.embedding_matrix,
            num_stocks=len(stocks),
            stock_emb_dim=HIDDEN_DIM,
            layer_dim=1,
            device=device
        ).to(device)

        mdgnn = MDGNN(
            stock_in_dim=4,
            bank_in_dim=4,
            industry_in_dim=1,
            edge_dims={"SB": 5, "BS": 5},
            hidden_dim=HIDDEN_DIM,
            gnn_layers=GNN_LAYERS,
            num_heads=4,
            ff_dim=256,
            dropout=0.1,
            task="regression"
        ).to(device)

        model = LSTM_MDGNN_Fusion(
            lstm_encoder=lstm_encoder,
            mdgnn_model=mdgnn,
            num_numeric_features=len(FEATURE_COLS),
            graph_hidden_dim=HIDDEN_DIM,
            hidden_dim=HIDDEN_DIM,
            dropout=0.1,
            task="regression",
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

        best_val_loss = float("inf")
        counter = 0
        save_path = f"best_fusion_model_split_{split_idx}.pt"

        for epoch in range(MAX_EPOCHS):
            # Training
            model.train()
            train_loss = 0.0

            progress_bar = tqdm.tqdm(
                train_loader,
                desc=f"Split {split_idx} Epoch {epoch + 1}/{MAX_EPOCHS}",
                leave=True
            )

            for X_text_batch, X_num_batch, Y_batch, stock_batch, graph_seq_batch in progress_bar:
                X_text_batch = X_text_batch.to(device)
                X_num_batch = X_num_batch.to(device)
                Y_batch = Y_batch.to(device)
                stock_batch = stock_batch.to(device)
                graph_seq_batch = graph_seq_batch.to(device)

                optimizer.zero_grad()

                preds = model(
                    text_ids=X_text_batch,
                    numeric_feats=X_num_batch,
                    stock_ids=stock_batch,
                    graph_seq=graph_seq_batch
                )

                loss = criterion(preds, Y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())


            # Validation
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for X_text_val, X_num_val, Y_val, stock_val, graph_seq_val in val_loader:
                    X_text_val = X_text_val.to(device)
                    X_num_val = X_num_val.to(device)
                    Y_val = Y_val.to(device)
                    stock_val = stock_val.to(device)
                    graph_seq_val = graph_seq_val.to(device)

                    preds = model(
                        text_ids=X_text_val,
                        numeric_feats=X_num_val,
                        stock_ids=stock_val,
                        graph_seq=graph_seq_val
                    )

                    val_loss += criterion(preds, Y_val).item()

            avg_train_loss = train_loss / max(len(train_loader), 1)
            avg_val_loss = val_loss / max(len(val_loader), 1)

            print(f"Split {split_idx} | Epoch {epoch + 1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                torch.save(model.state_dict(), save_path)
            else:
                counter += 1
                if counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break


        # Fixed-parameter prediction on following 6 months
        print(f"Testing split {split_idx}...")
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()

        test_loss = 0.0
        with torch.no_grad():
            for X_text_test, X_num_test, Y_test, stock_test, graph_seq_test in test_loader:
                X_text_test = X_text_test.to(device)
                X_num_test = X_num_test.to(device)
                Y_test = Y_test.to(device)
                stock_test = stock_test.to(device)
                graph_seq_test = graph_seq_test.to(device)

                preds = model(
                    text_ids=X_text_test,
                    numeric_feats=X_num_test,
                    stock_ids=stock_test,
                    graph_seq=graph_seq_test
                )

                test_loss += criterion(preds, Y_test).item()

        avg_test_loss = test_loss / max(len(test_loader), 1)
        all_test_losses.append(avg_test_loss)
        print(f"Split {split_idx} Test Loss={avg_test_loss:.4f}")

    print("\nAll rolling test losses:", all_test_losses)
    if len(all_test_losses) > 0:
        print("Average rolling test loss:", sum(all_test_losses) / len(all_test_losses))
    else:
        print("No valid rolling test results.")
