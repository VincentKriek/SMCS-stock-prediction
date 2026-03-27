from typing import Dict, List

import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import Dataset, DataLoader

from lstm import LazyHeadlineVectorizer, LSTM_Encoder
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
        text_embs = []

        for t in range(x_text_seq.size(1)):
            h_t = self.lstm_encoder(
                x_text_seq[:, t, :],
                stock_ids_seq[:, t]
            )
            text_embs.append(h_t)

        text_emb_seq = torch.stack(text_embs, dim=1)
        fused_seq = self.fuse(text_emb_seq, graph_emb_seq)
        pred = self.mdgnn.forward_from_sequence(fused_seq)
        return pred


WINDOW_SIZE = 30
BATCH_SIZE = 4
HIDDEN_DIM = 128
STOCK_EMB_DIM = 128
TASK = "regression"
MAX_EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 2

TEXT_N_ROWS = None
USE_ZERO_GRAPH_FALLBACK = True

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


def build_stock_key_candidates(x):
    if x is None:
        return []

    s = str(x).strip()
    candidates = [s, s.upper(), s.lower()]

    try:
        f = float(s)
        if f.is_integer():
            candidates.append(str(int(f)))
    except Exception:
        pass

    seen = set()
    result = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def find_graph_embedding(
    sid,
    dt,
    daily_stock_embeddings: Dict[pd.Timestamp, Dict[str, torch.Tensor]],
    graph_emb_dim: int
):
    if dt in daily_stock_embeddings:
        stock_map = daily_stock_embeddings[dt]
        for key in build_stock_key_candidates(sid):
            if key in stock_map:
                return stock_map[key]

    if USE_ZERO_GRAPH_FALLBACK:
        return torch.zeros(graph_emb_dim, dtype=torch.float32)

    return None


class RollingWindowDataset(Dataset):
    def __init__(
        self,
        rows: List[dict],
        stock2id: Dict[str, int],
        daily_stock_embeddings: Dict[pd.Timestamp, Dict[str, torch.Tensor]],
        max_headline_len: int,
        graph_emb_dim: int,
        window_size: int = 30,
        target_col: str = "close"
    ):
        self.rows = rows
        self.stock2id = stock2id
        self.daily_stock_embeddings = daily_stock_embeddings
        self.max_headline_len = max_headline_len
        self.graph_emb_dim = graph_emb_dim
        self.window_size = window_size
        self.target_col = target_col
        self.samples = []

        grouped: Dict[str, List[dict]] = {}
        for row in rows:
            sid = row.get("Stock_symbol")
            if sid is None:
                continue
            grouped.setdefault(sid, []).append(row)

        for sid, seq_rows in grouped.items():
            seq_rows = sorted(seq_rows, key=lambda r: to_timestamp(r["Date"]))

            if len(seq_rows) < window_size:
                continue

            for end_idx in range(window_size - 1, len(seq_rows)):
                window_rows = seq_rows[end_idx - window_size + 1:end_idx + 1]
                target_row = seq_rows[end_idx]

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

            if len(embedded) < self.max_headline_len:
                embedded = embedded + [0] * (self.max_headline_len - len(embedded))
            else:
                embedded = embedded[:self.max_headline_len]

            dt = to_timestamp(wr["Date"])
            graph_emb = find_graph_embedding(
                sid=sid,
                dt=dt,
                daily_stock_embeddings=self.daily_stock_embeddings,
                graph_emb_dim=self.graph_emb_dim
            )

            if graph_emb is None:
                graph_emb = torch.zeros(self.graph_emb_dim, dtype=torch.float32)

            x_text_seq.append(embedded)
            stock_ids_seq.append(stock_id_num)
            graph_emb_seq.append(graph_emb)

        x_text_seq = torch.tensor(x_text_seq, dtype=torch.long)
        stock_ids_seq = torch.tensor(stock_ids_seq, dtype=torch.long)
        graph_emb_seq = torch.stack(graph_emb_seq, dim=0).float()
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

l = LazyHeadlineVectorizer(
    "../prepared_data_2018-01-01_2023-12-31.parquet",
    n_rows=TEXT_N_ROWS
)
l.run()

if "Sentiment_llm_mean_filled" in l.lf.collect_schema().names():
    l.lf = l.lf.drop(
        "Sentiment_llm_mean_filled",
        "Sentiment_llm_median_filled",
        "Sentiment_llm_mode_filled"
    )

if "row_index" not in l.lf.collect_schema().names():
    l.lf = l.lf.with_row_count("row_index")

if "close" not in l.lf.collect_schema().names():
    raise ValueError("Column 'close' is missing from the dataset.")

stocks = sorted(l.lf.select("Stock_symbol").unique().collect().to_series().to_list())
stock2id = {symbol: idx for idx, symbol in enumerate(stocks)}
num_stocks = len(stock2id)

data = l.lf.sort(["Stock_symbol", "Date"]).collect(engine="streaming").to_dicts()

if len(data) == 0:
    raise ValueError("No rows were loaded from the dataset.")

nodes_bank_df = pd.read_parquet(BANK_NODES_PATH)
nodes_stock_df = pd.read_parquet(STOCK_NODES_PATH)
edges_df = pd.read_parquet(BANK_STOCK_EDGES_PATH)

quarter_snapshots = build_quarter_snapshots(nodes_bank_df, nodes_stock_df, edges_df)

all_trading_dates = sorted({to_timestamp(row["Date"]) for row in data})
daily_snapshots = expand_quarter_snapshots_to_daily(all_trading_dates, quarter_snapshots)

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

daily_stock_embeddings: Dict[pd.Timestamp, Dict[str, torch.Tensor]] = {}

mdgnn.eval()
with torch.no_grad():
    for dt, snapshot in tqdm.tqdm(daily_snapshots.items(), desc="Precomputing graph embeddings"):
        stock_repr = mdgnn.encode_snapshot(
            stock_feat=snapshot["stock_feat"].to(device),
            bank_feat=snapshot["bank_feat"].to(device),
            industry_feat=None,
            edges=snapshot["edges"]
        )

        stock_ids_snapshot = snapshot["stock_ids"]
        per_stock = {}
        for idx, sid in enumerate(stock_ids_snapshot):
            per_stock[str(sid).strip()] = stock_repr[idx].detach().cpu()

        daily_stock_embeddings[dt] = per_stock

graph_emb_dim = None
for _, mp in daily_stock_embeddings.items():
    if len(mp) > 0:
        graph_emb_dim = next(iter(mp.values())).shape[-1]
        break

if graph_emb_dim is None:
    raise ValueError("Could not infer graph embedding dimension.")

train_rows, val_rows, test_rows = split_rows_by_date(data, train_ratio=0.8, val_ratio=0.1)

train_dataset = RollingWindowDataset(
    rows=train_rows,
    stock2id=stock2id,
    daily_stock_embeddings=daily_stock_embeddings,
    max_headline_len=l.max_headline_len,
    graph_emb_dim=graph_emb_dim,
    window_size=WINDOW_SIZE,
    target_col="close"
)

val_dataset = RollingWindowDataset(
    rows=val_rows,
    stock2id=stock2id,
    daily_stock_embeddings=daily_stock_embeddings,
    max_headline_len=l.max_headline_len,
    graph_emb_dim=graph_emb_dim,
    window_size=WINDOW_SIZE,
    target_col="close"
)

test_dataset = RollingWindowDataset(
    rows=test_rows,
    stock2id=stock2id,
    daily_stock_embeddings=daily_stock_embeddings,
    max_headline_len=l.max_headline_len,
    graph_emb_dim=graph_emb_dim,
    window_size=WINDOW_SIZE,
    target_col="close"
)

if len(train_dataset) == 0:
    raise ValueError("train_dataset is empty.")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) if len(val_dataset) > 0 else None
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) if len(test_dataset) > 0 else None

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

for epoch in range(MAX_EPOCHS):
    model.train()
    train_loss = 0.0

    progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{MAX_EPOCHS}", leave=True)
    for x_text_seq, stock_ids_seq, graph_emb_seq, y_batch, _ in progress_bar:
        x_text_seq = x_text_seq.to(device)
        stock_ids_seq = stock_ids_seq.to(device)
        graph_emb_seq = graph_emb_seq.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        preds = model(x_text_seq, stock_ids_seq, graph_emb_seq)
        loss = criterion(preds, y_batch)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / max(len(train_loader), 1)

    if val_loader is not None:
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

        avg_val_loss = val_loss / max(len(val_loader), 1)
        print(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            counter += 1
            if counter >= PATIENCE:
                break
    else:
        print(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}")
        torch.save(model.state_dict(), "best_model.pt")

model.load_state_dict(torch.load("best_model.pt", map_location=device))

all_row_idxs = []
all_preds = []

if test_loader is not None:
    model.eval()
    with torch.no_grad():
        for x_text_seq, stock_ids_seq, graph_emb_seq, _, row_idx in test_loader:
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
    print(lf_joined.collect())
