from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import Dataset, DataLoader

from src.lstm import LazyHeadlineVectorizer, LSTM_Encoder
from Mdgnn import MDGNN
from graph_snapshot import build_quarter_snapshots, expand_quarter_snapshots_to_daily


class LSTM_MDGNN_Fusion(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        embedding_matrix,
        num_stocks,
        stock_emb_dim,
        mdgnn: MDGNN,
        layer_dim=1,
        device="cpu",
        fusion="concat"
    ):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.fusion = fusion
        self.mdgnn = mdgnn

        self.lstm_encoder = LSTM_Encoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            embedding_matrix=embedding_matrix,
            num_stocks=num_stocks,
            stock_emb_dim=stock_emb_dim,
            layer_dim=layer_dim,
            device=device
        )

        if fusion == "concat":
            self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        elif fusion == "add":
            self.fusion_proj = None
        else:
            raise ValueError("fusion must be 'concat' or 'add'")

    def fuse(self, text_emb, graph_emb):
        if self.fusion == "concat":
            return self.fusion_proj(torch.cat([text_emb, graph_emb], dim=-1))
        return text_emb + graph_emb

    def forward(self, x_text_seq, stock_ids_seq, graph_emb_seq):
        B, T, L = x_text_seq.shape
        text_embs = []

        for t in range(T):
            h_t = self.lstm_encoder(
                x_text_seq[:, t, :],
                stock_ids_seq[:, t]
            )  # [B, D]
            text_embs.append(h_t)

        text_emb_seq = torch.stack(text_embs, dim=1)  # [B, T, D]
        fused_seq = self.fuse(text_emb_seq, graph_emb_seq)
        pred = self.mdgnn.forward_from_sequence(fused_seq)
        return pred



WINDOW_SIZE = 180
BATCH_SIZE = 4
HIDDEN_DIM = 128
STOCK_EMB_DIM = 128  
TASK = "regression"
MAX_EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 2

# File paths for graph parquet files
BANK_NODES_PATH = "nodes_bank.parquet"
STOCK_NODES_PATH = "nodes_stock.parquet"
BANK_STOCK_EDGES_PATH = "edges_bank_stock.parquet"



def safe_float(x, default=0.0):
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def to_timestamp(x):
    return pd.Timestamp(x)


# Load + vectorize text data

l = LazyHeadlineVectorizer("prepared_data_2018-01-01_2023-12-31.parquet", n_rows=10)
l.run()

if "Sentiment_llm_mean_filled" in l.lf.collect_schema().names():
    l.lf = l.lf.drop("Sentiment_llm_mean_filled", "Sentiment_llm_median_filled", "Sentiment_llm_mode_filled")

print("\n===== Schema after vectorizing headlines =====:")
print(l.lf.collect().schema)
print()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(f"Max headline len: {l.max_headline_len}")
print("=" * 40)
print("Loading Data")

# stock mappings from the news/market data side
stocks = sorted(l.lf.select("Stock_symbol").unique().collect().to_series().to_list())
stock2id = {symbol: idx for idx, symbol in enumerate(stocks)}
id2stock = {idx: symbol for symbol, idx in stock2id.items()}
num_stocks = len(stock2id)

data = l.lf.sort(["Stock_symbol", "Date"]).collect(engine="streaming").to_dicts()



# Graph preparation
print("=" * 40)
print("Loading Graph Data")

nodes_bank_df = pd.read_parquet(BANK_NODES_PATH)
nodes_stock_df = pd.read_parquet(STOCK_NODES_PATH)
edges_df = pd.read_parquet(BANK_STOCK_EDGES_PATH)

quarter_snapshots = build_quarter_snapshots(nodes_bank_df, nodes_stock_df, edges_df)

# trading dates from text/market data
all_trading_dates = sorted({to_timestamp(row["Date"]) for row in data})
daily_snapshots = expand_quarter_snapshots_to_daily(all_trading_dates, quarter_snapshots)

print(f"Quarter snapshots: {len(quarter_snapshots)}")
print(f"Daily snapshot mappings: {len(daily_snapshots)}")



# Build MDGNN + precompute daily stock graph embeddings
print("=" * 40)
print("Building MDGNN and precomputing daily graph embeddings")

edge_dims = {
    "SS": 0,
    "SB": 5,
    "BS": 5,
    "SI": 0,
    "IS": 0,
    "II": 0,
}

mdgnn = MDGNN(
    stock_in_dim=4,      
    bank_in_dim=4,       
    industry_in_dim=1,   
    edge_dims=edge_dims,
    hidden_dim=HIDDEN_DIM,
    gnn_layers=2,
    num_heads=4,
    ff_dim=256,
    dropout=0.1,
    alibi_slope=1.0,
    task=TASK
).to(device)

# Precompute graph embeddings for each daily snapshot.
# Same quarter => same graph => same stock embeddings reused for all days in that quarter.
daily_stock_embeddings: Dict[pd.Timestamp, Dict[str, torch.Tensor]] = {}

mdgnn.eval()
with torch.no_grad():
    for dt, snapshot in tqdm.tqdm(daily_snapshots.items(), desc="Precomputing graph embeddings"):
        stock_feat = snapshot["stock_feat"].to(device)
        bank_feat = snapshot["bank_feat"].to(device)
        edges = {
            k: (e[0].to(device), e[1].to(device) if e[1] is not None else None)
            if e is not None else None
            for k, e in snapshot["edges"].items()
        }

        stock_repr = mdgnn.encode_snapshot(
            stock_feat=stock_feat,
            bank_feat=bank_feat,
            industry_feat=None,
            edges=edges
        )  # [Ns, D]

        stock_ids_snapshot = snapshot["stock_ids"]
        per_stock = {}
        for idx, sid in enumerate(stock_ids_snapshot):
            per_stock[str(sid)] = stock_repr[idx].detach().cpu()

        daily_stock_embeddings[dt] = per_stock



# Build rolling-window samples
class RollingWindowDataset(Dataset):
    """
    Each sample:
        x_text_seq:    [T, L]
        stock_ids_seq: [T]
        graph_emb_seq: [T, D]
        target:        scalar
        row_idx:       original row index of target day
    """
    def __init__(
        self,
        rows: List[dict],
        stock2id: Dict[str, int],
        daily_stock_embeddings: Dict[pd.Timestamp, Dict[str, torch.Tensor]],
        max_headline_len: int,
        window_size: int = 180,
        target_col: str = "close"
    ):
        self.rows = rows
        self.stock2id = stock2id
        self.daily_stock_embeddings = daily_stock_embeddings
        self.max_headline_len = max_headline_len
        self.window_size = window_size
        self.target_col = target_col
        self.samples = []

        # group by stock
        grouped: Dict[str, List[dict]] = {}
        for row in rows:
            sid = row.get("Stock_symbol")
            if sid is None:
                continue
            grouped.setdefault(sid, []).append(row)

        for sid, seq_rows in grouped.items():
            seq_rows = sorted(seq_rows, key=lambda r: to_timestamp(r["Date"]))

            # only keep windows with enough history
            for end_idx in range(window_size - 1, len(seq_rows)):
                window_rows = seq_rows[end_idx - window_size + 1 : end_idx + 1]
                target_row = seq_rows[end_idx]

                # check all dates have graph embedding for this stock
                valid = True
                for wr in window_rows:
                    dt = to_timestamp(wr["Date"])
                    if dt not in daily_stock_embeddings:
                        valid = False
                        break
                    if str(sid) not in daily_stock_embeddings[dt]:
                        valid = False
                        break

                if not valid:
                    continue

                target = target_row.get(target_col)
                if target is None:
                    continue

                self.samples.append((sid, window_rows, target_row))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, window_rows, target_row = self.samples[idx]

        x_text_seq = []
        stock_ids_seq = []
        graph_emb_seq = []

        stock_id_num = self.stock2id[sid]

        for wr in window_rows:
            embedded = wr.get("embedded_headline")
            if embedded is None:
                embedded = [0] * self.max_headline_len

            # make sure length is fixed
            if len(embedded) < self.max_headline_len:
                embedded = embedded + [0] * (self.max_headline_len - len(embedded))
            else:
                embedded = embedded[:self.max_headline_len]

            dt = to_timestamp(wr["Date"])
            graph_emb = self.daily_stock_embeddings[dt][str(sid)]  # [D]

            x_text_seq.append(embedded)
            stock_ids_seq.append(stock_id_num)
            graph_emb_seq.append(graph_emb)

        x_text_seq = torch.tensor(x_text_seq, dtype=torch.long)                    # [T, L]
        stock_ids_seq = torch.tensor(stock_ids_seq, dtype=torch.long)              # [T]
        graph_emb_seq = torch.stack(graph_emb_seq, dim=0).float()                  # [T, D]
        target = torch.tensor(safe_float(target_row[self.target_col]), dtype=torch.float32)
        row_idx = torch.tensor(int(target_row.get("row_index", -1)), dtype=torch.long)

        return x_text_seq, stock_ids_seq, graph_emb_seq, target, row_idx


def split_rows_by_date(rows: List[dict], train_ratio=0.8, val_ratio=0.1):
    unique_dates = sorted({to_timestamp(r["Date"]) for r in rows})
    n_dates = len(unique_dates)

    train_cut = int(n_dates * train_ratio)
    val_cut = int(n_dates * (train_ratio + val_ratio))

    train_dates = set(unique_dates[:train_cut])
    val_dates = set(unique_dates[train_cut:val_cut])
    test_dates = set(unique_dates[val_cut:])

    train_rows = [r for r in rows if to_timestamp(r["Date"]) in train_dates]
    val_rows = [r for r in rows if to_timestamp(r["Date"]) in val_dates]
    test_rows = [r for r in rows if to_timestamp(r["Date"]) in test_dates]

    return train_rows, val_rows, test_rows


train_rows, val_rows, test_rows = split_rows_by_date(data, train_ratio=0.8, val_ratio=0.1)

train_dataset = RollingWindowDataset(
    rows=train_rows,
    stock2id=stock2id,
    daily_stock_embeddings=daily_stock_embeddings,
    max_headline_len=l.max_headline_len,
    window_size=WINDOW_SIZE,
    target_col="close"
)

val_dataset = RollingWindowDataset(
    rows=val_rows,
    stock2id=stock2id,
    daily_stock_embeddings=daily_stock_embeddings,
    max_headline_len=l.max_headline_len,
    window_size=WINDOW_SIZE,
    target_col="close"
)

test_dataset = RollingWindowDataset(
    rows=test_rows,
    stock2id=stock2id,
    daily_stock_embeddings=daily_stock_embeddings,
    max_headline_len=l.max_headline_len,
    window_size=WINDOW_SIZE,
    target_col="close"
)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples:   {len(val_dataset)}")
print(f"Test samples:  {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Build fusion model
print("=" * 40)
print("Building fusion model")

model = LSTM_MDGNN_Fusion(
    vocab_size=len(l.word2id),
    embedding_dim=l.vector_size,
    hidden_dim=HIDDEN_DIM,
    embedding_matrix=l.embedding_matrix,
    num_stocks=num_stocks,
    stock_emb_dim=STOCK_EMB_DIM,
    mdgnn=mdgnn,
    layer_dim=1,
    device=device,
    fusion="concat"
).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

best_val_loss = float("inf")
counter = 0



# Train 
print("=" * 40)
print("Training-Val")

for epoch in range(MAX_EPOCHS):
    model.train()
    train_loss = 0.0

    progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=True)
    for x_text_seq, stock_ids_seq, graph_emb_seq, y_batch, _ in progress_bar:
        x_text_seq = x_text_seq.to(device)          # [B, T, L]
        stock_ids_seq = stock_ids_seq.to(device)    # [B, T]
        graph_emb_seq = graph_emb_seq.to(device)    # [B, T, D]
        y_batch = y_batch.to(device)                # [B]

        optimizer.zero_grad()

        preds = model(x_text_seq, stock_ids_seq, graph_emb_seq)  # [B]
        loss = criterion(preds, y_batch)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_text_seq, stock_ids_seq, graph_emb_seq, y_batch, _ in val_loader:
            x_text_seq = x_text_seq.to(device)
            stock_ids_seq = stock_ids_seq.to(device)
            graph_emb_seq = graph_emb_seq.to(device)
            y_batch = y_batch.to(device)

            preds = model(x_text_seq, stock_ids_seq, graph_emb_seq)
            val_loss += criterion(preds, y_batch).item()

    avg_train_loss = train_loss / max(len(train_loader), 1)
    avg_val_loss = val_loss / max(len(val_loader), 1)

    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        counter += 1
        if counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

model.load_state_dict(torch.load("best_model.pt", map_location=device))



# Test
print("=" * 40)
print("Testing")

all_row_idxs = []
all_preds = []

model.eval()
with torch.no_grad():
    for x_text_seq, stock_ids_seq, graph_emb_seq, y_batch, row_idx in test_loader:
        x_text_seq = x_text_seq.to(device)
        stock_ids_seq = stock_ids_seq.to(device)
        graph_emb_seq = graph_emb_seq.to(device)

        preds = model(x_text_seq, stock_ids_seq, graph_emb_seq)

        for idx, pred in zip(row_idx.cpu(), preds.cpu()):
            if int(idx) != -1:
                all_row_idxs.append(int(idx))
                all_preds.append(float(pred))

if len(all_row_idxs) > 0:
    preds_lf = pl.LazyFrame({
        "row_index": all_row_idxs,
        "prediction": all_preds
    })

    original_lf = l.lf
    lf_joined = original_lf.join(preds_lf, on="row_index", how="inner")

    print("\n===== Predictions joined with original data =====")
    print(lf_joined.collect())
else:
    print("No predictions were collected.")