from lstm import LazyHeadlineVectorizer, LSTM_Encoder
from Mdgnn import MDGNN
from graph_snapshot import build_quarter_snapshots, expand_quarter_snapshots_to_daily

import polars as pl
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import r2_score


# Configuration
NEWS_N_ROWS = None

ROLLING_START_DATE = "2018-01-01"
ROLLING_END_DATE = "2023-02-28"
TRAIN_MONTHS = 6
VAL_MONTHS = 1
TEST_MONTHS = 6

HIDDEN_DIM = 128
GNN_LAYERS = 2
WINDOW_SIZE = 10
MAX_EPOCHS = 500
PATIENCE = 20
BATCH_SIZE = 8

# For lstm:
FEATURE_COLS = [
    "open",
    "high",
    "low",
    "close",
    "adj close",
    "volume",
    "Sentiment_llm_mean_filled",
]

TARGET_COL = "target_return"


class DailyGraphFeatureCache:
    def __init__(
        self,
        mdgnn_model: MDGNN,
        nodes_bank_path: str,
        nodes_stock_path: str,
        edges_bank_stock_path: str,
        edges_stock_stock_path: str,
        trading_dates,
        device: torch.device,
        hidden_dim: int = 128,
    ):
        self.device = device
        self.hidden_dim = hidden_dim
        self.cache = {}

        # 1. Determine needed quarters
        trading_ts = [pd.Timestamp(x) for x in trading_dates]
        needed_quarters = sorted(
            {(ts.year, ((ts.month - 1) // 3 + 1)) for ts in trading_ts}
        )

        quarter_expr = None
        for y, q in needed_quarters:
            cond = (pl.col("year") == y) & (pl.col("quarter") == q)
            quarter_expr = cond if quarter_expr is None else (quarter_expr | cond)

        if quarter_expr is None:
            self.sorted_dates = []
            return

        # 2. Read only required node columns
        stock_cols = [
            "year",
            "quarter",
            "stock_id",
            "num_holders",
            "total_institutional_value",
            "total_institutional_shares",
            "num_quarters_held",
        ]
        bank_cols = [
            "year",
            "quarter",
            "bank_id",
            "num_stocks_held",
            "total_aum_value",
            "avg_position_size",
            "num_quarters_active",
        ]

        print("Loading stock node data...")
        nodes_stock_df = (
            pl.scan_parquet(nodes_stock_path)
            .select(stock_cols)
            .filter(quarter_expr)
            .collect(engine="streaming")
            .to_pandas()
        )

        print("Loading bank node data...")
        nodes_bank_df = (
            pl.scan_parquet(nodes_bank_path)
            .select(bank_cols)
            .filter(quarter_expr)
            .collect(engine="streaming")
            .to_pandas()
        )

        if len(nodes_stock_df) == 0 or len(nodes_bank_df) == 0:
            self.sorted_dates = []
            return

        # 3. Read only required edge columns
        edge_bs_cols = [
            "year",
            "quarter",
            "bank_id",
            "stock_id",
            "total_value",
            "total_shares",
            "voting_sole",
            "voting_shared",
            "voting_none",
        ]
        edge_ss_cols = [
            "year",
            "quarter",
            "stock_id_1",
            "stock_id_2",
            "co_holder_count",
        ]

        print("Loading bank-stock edge data...")
        edges_bank_stock_df = (
            pl.scan_parquet(edges_bank_stock_path)
            .select(edge_bs_cols)
            .filter(quarter_expr)
            .collect(engine="streaming")
            .to_pandas()
        )

        print("Loading stock-stock edge data...")
        edges_stock_stock_df = (
            pl.scan_parquet(edges_stock_stock_path)
            .select(edge_ss_cols)
            .filter(quarter_expr)
            .collect(engine="streaming")
            .to_pandas()
        )

        # 4. Build quarter snapshots
        print("Building quarter snapshots...")
        quarter_snapshots = build_quarter_snapshots(
            nodes_bank_df=nodes_bank_df,
            nodes_stock_df=nodes_stock_df,
            edges_bank_stock_df=edges_bank_stock_df,
            edges_stock_stock_df=edges_stock_stock_df,
        )

        print("Expanding quarter snapshots to daily snapshots...")
        daily_snapshots = expand_quarter_snapshots_to_daily(
            trading_dates=trading_dates,
            quarter_snapshots=quarter_snapshots,
        )

        mdgnn_model = mdgnn_model.to(device)
        mdgnn_model.eval()

        print("Encoding graph snapshots...")
        with torch.no_grad():
            for dt, snap in tqdm(
                daily_snapshots.items(),
                desc="Processing days",
                total=len(daily_snapshots),
            ):
                stock_feat = snap["stock_feat"].to(device)

                bank_feat = snap["bank_feat"]
                if bank_feat is not None:
                    bank_feat = bank_feat.to(device)

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
                            edge_attr.to(device) if edge_attr is not None else None,
                        )

                stock_emb = mdgnn_model.encode_snapshot(
                    stock_feat=stock_feat,
                    bank_feat=bank_feat,
                    industry_feat=industry_feat,
                    edges=edges,
                )

                pooled = stock_emb.mean(dim=0).detach().cpu()
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
            pads = [
                torch.zeros(self.hidden_dim, dtype=torch.float32)
                for _ in range(pad_len)
            ]
            seq = pads + seq

        return torch.stack(seq, dim=0)


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
        window_size: int = 10,
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

            self.samples.append(
                {
                    "text_ids": torch.tensor(text_ids, dtype=torch.long),
                    "numeric_feats": torch.tensor(numeric_feats, dtype=torch.float32),
                    "target": torch.tensor(float(target), dtype=torch.float32),
                    "stock_id": torch.tensor(stock2id[stock_symbol], dtype=torch.long),
                    "graph_seq": (
                        graph_seq
                        if graph_seq is not None
                        else torch.zeros(
                            window_size, graph_hidden_dim, dtype=torch.float32
                        )
                    ),
                }
            )

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


class LSTM_MDGNN_Fusion(nn.Module):
    def __init__(
        self,
        lstm_encoder: LSTM_Encoder,
        mdgnn_model: MDGNN,
        num_numeric_features: int,
        graph_hidden_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.lstm_encoder = lstm_encoder
        self.mdgnn_model = mdgnn_model

        self.numeric_proj = nn.Sequential(
            nn.Linear(num_numeric_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.graph_proj = nn.Sequential(
            nn.Linear(graph_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text_ids, numeric_feats, stock_ids, graph_seq):
        text_repr = self.lstm_encoder(text_ids, stock_ids)
        num_repr = self.numeric_proj(numeric_feats)

        _, graph_repr, _ = self.mdgnn_model.forward(
            graph_seq, return_attention=True
        )
        graph_repr = self.graph_proj(graph_repr)

        fused = torch.cat([text_repr, num_repr, graph_repr], dim=1)
        out = self.fusion_head(fused).squeeze(-1)
        return out


def add_next_day_return_target(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = lf.sort(["Stock_symbol", "Date"]).with_columns(
        (
            (pl.col("close").shift(-1).over("Stock_symbol") - pl.col("close"))
            / pl.col("close")
        ).alias("target_return")
    )
    lf = lf.filter(pl.col("target_return").is_not_null())
    return lf


def make_halfyear_rolling_splits(
    data: pl.LazyFrame,
    start_date: str = "2018-01-01",
    end_date: str = "2023-02-28",
    train_months: int = 6,
    val_months: int = 1,
    test_months: int = 6,
):
    schema_names = data.collect_schema().names()
    needed_cols = [
        "Date",
        "Stock_symbol",
        "embedded_headline",
        "open",
        "high",
        "target_return",
    ]
    used_cols = [c for c in needed_cols if c in schema_names]

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    print("Building rolling splits...")
    df = (
        data.select(used_cols)
        .filter(
            (pl.col("Date") >= pl.lit(start_dt)) & (pl.col("Date") <= pl.lit(end_dt))
        )
        .sort("Date")
        .collect(engine="streaming")
        .to_pandas()
    )

    if len(df) == 0:
        return []

    df["Date"] = pd.to_datetime(df["Date"])

    splits = []
    anchor = start_dt

    while True:
        train_start = anchor
        train_end = (
            train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        )

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
            splits.append(
                {
                    "train_rows": train_df.to_dict("records"),
                    "val_rows": val_df.to_dict("records"),
                    "test_rows": test_df.to_dict("records"),
                }
            )

        anchor = anchor + pd.DateOffset(months=test_months)
        if anchor > end_dt:
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
        graph_hidden_dim=HIDDEN_DIM,
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
        graph_hidden_dim=HIDDEN_DIM,
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
        graph_hidden_dim=HIDDEN_DIM,
        window_size=window_size,
    )

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


if __name__ == "__main__":
    lhv = LazyHeadlineVectorizer(
        "data/pre-processor/prepared_data_2018-01-01_2023-12-31.parquet",
        n_rows=NEWS_N_ROWS,
    )
    lhv.run()

    drop_cols = [
        # "Sentiment_llm_mean_filled",
        "Sentiment_llm_median_filled",
        "Sentiment_llm_mode_filled",
    ]
    schema_names = lhv.lf.collect_schema().names()
    existing_drop_cols = [c for c in drop_cols if c in schema_names]
    if existing_drop_cols:
        lhv.lf = lhv.lf.drop(*existing_drop_cols)

    lhv.lf = add_next_day_return_target(lhv.lf)
    print("=== Added next day return column ===")
    print(lhv.lf.collect_schema().keys())

    stocks = sorted(
        lhv.lf.select("Stock_symbol").unique().collect().to_series().to_list()
    )
    stock2id = {symbol: idx for idx, symbol in enumerate(stocks)}

    cache_device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"CACHE DEVICE: {cache_device}")
    print(f"DEVICE: {device}")

    mdgnn_for_cache = MDGNN(
        stock_in_dim=4,
        bank_in_dim=4,
        industry_in_dim=1,
        edge_dims={"SS": 1, "SB": 5, "BS": 5},
        hidden_dim=HIDDEN_DIM,
        gnn_layers=GNN_LAYERS,
        num_heads=4,
        ff_dim=256,
        dropout=0.1,
    ).to(cache_device)

    trading_dates = (
        lhv.lf.select("Date").unique().sort("Date").collect().to_series().to_list()
    )

    print("Building full graph cache...")
    graph_cache = DailyGraphFeatureCache(
        mdgnn_model=mdgnn_for_cache,
        nodes_bank_path="data/graphs/nodes_bank.parquet",
        nodes_stock_path="data/graphs/nodes_stock.parquet",
        edges_bank_stock_path="data/graphs/edges_bank_stock.parquet",
        edges_stock_stock_path="data/graphs/edges_stock_stock.parquet",
        trading_dates=trading_dates,
        device=cache_device,
        hidden_dim=HIDDEN_DIM,
    )

    # Save cache
    torch.save(graph_cache.cache, "graph_cache.pt")

    # Load cache later
    loaded_cache = torch.load("graph_cache.pt")
    graph_cache.cache = loaded_cache
    graph_cache.sorted_dates = sorted(loaded_cache.keys())

    rolling_splits = make_halfyear_rolling_splits(
        data=lhv.lf,
        start_date=ROLLING_START_DATE,
        end_date=ROLLING_END_DATE,
        train_months=TRAIN_MONTHS,
        val_months=VAL_MONTHS,
        test_months=TEST_MONTHS,
    )

    if len(rolling_splits) == 0:
        print("No valid rolling splits were created.")
        raise SystemExit

    for split_idx, split_info in enumerate(rolling_splits, start=1):
        print(f"Starting split {split_idx}...")

        train_loader, val_loader, test_loader = build_loaders_for_split(
            split_dict=split_info,
            stock2id=stock2id,
            graph_cache=graph_cache,
            batch_size=BATCH_SIZE,
            max_headline_len=lhv.max_headline_len,
            feature_cols=FEATURE_COLS,
            target_col=TARGET_COL,
            window_size=WINDOW_SIZE,
        )

        if (
            len(train_loader.dataset) == 0
            or len(val_loader.dataset) == 0
            or len(test_loader.dataset) == 0
        ):
            print(f"Split {split_idx} is empty. Skipping.")
            continue

        lstm_encoder = LSTM_Encoder(
            vocab_size=len(lhv.word2id),
            embedding_dim=lhv.vector_size,
            hidden_dim=HIDDEN_DIM,
            embedding_matrix=lhv.embedding_matrix,
            num_stocks=len(stocks),
            stock_emb_dim=HIDDEN_DIM,
            layer_dim=1,
            device=device,
        ).to(device)

        mdgnn = MDGNN(
            stock_in_dim=4,
            bank_in_dim=4,
            industry_in_dim=1,
            edge_dims={"SS": 1, "SB": 5, "BS": 5},
            hidden_dim=HIDDEN_DIM,
            gnn_layers=GNN_LAYERS,
            num_heads=4,
            ff_dim=256,
            dropout=0.1,
        ).to(device)

        model = LSTM_MDGNN_Fusion(
            lstm_encoder=lstm_encoder,
            mdgnn_model=mdgnn,
            num_numeric_features=len(FEATURE_COLS),
            graph_hidden_dim=HIDDEN_DIM,
            hidden_dim=HIDDEN_DIM,
            dropout=0.1,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

        best_val_loss = float("inf")
        counter = 0
        save_path = f"data/model/best_fusion_model_split_{split_idx}.pt"

        train_losses = []
        val_losses = []

        for epoch in range(MAX_EPOCHS):
            # region training / validating
            model.train()
            train_loss = 0.0

            for (
                X_text_batch,
                X_num_batch,
                Y_batch,
                stock_batch,
                graph_seq_batch,
            ) in train_loader:
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
                    graph_seq=graph_seq_batch,
                )

                loss = criterion(preds, Y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (
                    X_text_val,
                    X_num_val,
                    Y_val,
                    stock_val,
                    graph_seq_val,
                ) in val_loader:
                    X_text_val = X_text_val.to(device)
                    X_num_val = X_num_val.to(device)
                    Y_val = Y_val.to(device)
                    stock_val = stock_val.to(device)
                    graph_seq_val = graph_seq_val.to(device)

                    preds = model(
                        text_ids=X_text_val,
                        numeric_feats=X_num_val,
                        stock_ids=stock_val,
                        graph_seq=graph_seq_val,
                    )
                    val_loss += criterion(preds, Y_val).item()

            avg_train_loss = train_loss / max(len(train_loader), 1)
            avg_val_loss = val_loss / max(len(val_loader), 1)

            print(
                f"Split {split_idx} Epoch {epoch + 1} | Train={avg_train_loss:.6f} | Val={avg_val_loss:.6f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                torch.save(model.state_dict(), save_path)
            else:
                counter += 1
                if counter >= PATIENCE:
                    break

        loss_df = pd.DataFrame(
            {
                "epoch": list(range(1, len(train_losses) + 1)),
                "train_loss": train_losses,
                "val_loss": val_losses,
            }
        )

        loss_df.to_csv(f"data/model/losses_split_{split_idx}.csv", index=False)

        # region testing
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()

        test_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for (
                X_text_test,
                X_num_test,
                Y_test,
                stock_test,
                graph_seq_test,
            ) in test_loader:
                X_text_test = X_text_test.to(device)
                X_num_test = X_num_test.to(device)
                Y_test = Y_test.to(device)
                stock_test = stock_test.to(device)
                graph_seq_test = graph_seq_test.to(device)

                preds = model(
                    text_ids=X_text_test,
                    numeric_feats=X_num_test,
                    stock_ids=stock_test,
                    graph_seq=graph_seq_test,
                )

                test_loss += criterion(preds, Y_test).item()
                all_preds.extend(preds.detach().cpu().tolist())
                all_targets.extend(Y_test.detach().cpu().tolist())

        avg_test_loss = test_loss / max(len(test_loader), 1)

        r2 = r2_score(all_targets, all_preds)

        print(f"Split {split_idx} | Test={avg_test_loss:.6f}")
        print(f"Split {split_idx} | R2={r2:.6f}")
        print(f"Split {split_idx} | Predictions={all_preds}")

        df = pd.DataFrame({"prediction": all_preds, "target": all_targets})
        df.to_csv(f"data/model/predictions_split_{split_idx}.csv", index=False)
