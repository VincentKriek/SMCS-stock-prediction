from lstm import LazyHeadlineVectorizer, LSTM_Encoder
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
import tqdm
from datetime import datetime

# start = datetime(2018, 1, 1)
# end = datetime(2018, 2, 1)
# l = LazyHeadlineVectorizer("../news_formatted_2018-01-01_2023-12-31.parquet", start_date=start, end_date=end)
# l = LazyHeadlineVectorizer("../news_formatted_2018-01-01_2023-12-31.parquet", n_rows=1_000)
l = LazyHeadlineVectorizer("../prepared_data_2018-01-01_2023-12-31.parquet", n_rows=20)

# l.load_headlines()

# print(l.lf.collect().schema) # TODO: is Article_tile a List[str]? or only a str?
# print(l.lf.sort("Date").collect()) # TODO: is Article_tile a List[str]? or only a str?

l.run()
l.lf = l.lf.drop("Sentiment_llm_mean_filled", "Sentiment_llm_median_filled", "Sentiment_llm_mode_filled")

print("\n===== Schema after vectorizing headlines =====:")
print(l.lf.collect().schema) # TODO: is Article_tile a List[str]? or only a str?
print()

num_stocks = l.lf.select(pl.col("Stock_symbol").n_unique()).collect().item()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm = LSTM_Encoder(
    vocab_size=len(l.word2id),
    embedding_dim=l.vector_size,
    hidden_dim=128,
    embedding_matrix=l.embedding_matrix,
    num_stocks=num_stocks,
    stock_emb_dim=128,
    layer_dim = 1,
    device=device
)

print(f"Max headline len: {l.max_headline_len}")
# print(l.lf.sort("Date").drop("summary", "high", "low", "volume", "adj close", "headline_len", "tokenized_headline").collect())

# region Training

# Map stock symbols (string) to ints:
print("="*40)
print("Loading Data")
word2id = {word: idx for word, idx in l.word2id.items()}
id2word = {idx: word for word, idx in l.word2id.items()}

stocks = sorted(l.lf.select("Stock_symbol").unique().collect().to_series().to_list())
stock2id = {symbol: idx for idx, symbol in enumerate(stocks)}
id2stock = {idx: symbol for idx, symbol in enumerate(stocks)}

def build_dataset(
    data,
    stock2id: dict[str, int],
    embedding_col: str|None, # e.g. "embedded_headlines"
    feature_cols: list[str],
    target_col: str, # e.g. close price
    max_headline_len: int
):
    X_text_list, X_num_list, Y_list, stock_ids_list, row_idx_list = [], [], [], [], []

    for row in data:
        embedding_headline = []
        val = row.get(embedding_col)
        if val is None:
            val = [0] * max_headline_len # zero vector if there is no news that day
        embedding_headline.extend(val)

        numeric_feats = []
        for col in feature_cols:
            val = row.get(col)
            # Handle missing values
            if val is None:
                val = -1
            numeric_feats.append(val)

        target = row.get(target_col)
        if target is None:
            continue # skip if no label

        X_text_list.append(embedding_headline)
        X_num_list.append(numeric_feats)
        Y_list.append(target)
        stock_ids_list.append(stock2id[row["Stock_symbol"]])
        row_idx = row["row_index"][0] if row["row_index"] is not None else -1 # TODO: valid? use [0], as we've concatenated the headlines of these rows already?
        row_idx_list.append(row_idx)

    X_text = torch.tensor(X_text_list, dtype=torch.float32) # shape: (#rows, #max_headline_len)
    X_num = torch.tensor(X_num_list, dtype=torch.float32) # shape: (#rows, #features)
    print(X_text.shape)
    print(X_num.shape)
    Y = torch.tensor(Y_list, dtype=torch.float32) # shape: (#rows,)
    stock_ids = torch.tensor(stock_ids_list, dtype=torch.long)
    row_idx_tensor = torch.tensor(row_idx_list, dtype=torch.long)

    return TensorDataset(X_text, X_num, Y, stock_ids, row_idx_tensor)

data = l.lf.sort("Date")

feature_cols = ["open", "high"]

dataset = build_dataset(
    data=data.collect(engine="streaming").to_dicts(),
    stock2id=stock2id,
    embedding_col="embedded_headline",
    feature_cols=feature_cols,
    target_col="close",
    max_headline_len=l.max_headline_len
)

print("\nDataset:")
for t in dataset.tensors:
    print(t)

def train_val_test_split_ratio(
    data: pl.LazyFrame,
    batch_size: int,
    max_headline_l: int,
    train_ratio=0.8,
    val_ratio=0.1,
):
    # Split into train and test
    split_date = data.select("Date").collect()[int(train_ratio * l.n_rows)].item()
    train_data = data.filter(pl.col("Date") <= split_date)
    test_data = data.filter(pl.col("Date") > split_date)

    # Training-Validation
    train_data_full = train_data.collect(engine="streaming").to_dicts()
    split_idx = int(len(train_data_full) * (1 - val_ratio))
    train_data_final = train_data_full[:split_idx]
    val_data = train_data_full[split_idx:]

    e_col = "embedded_headline"
    f_cols = ["open", "high"]
    t_col = "close"

    train_loader = DataLoader(build_dataset(train_data_final, stock2id, e_col, f_cols, t_col, max_headline_l), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(build_dataset(val_data, stock2id, e_col, f_cols, t_col, max_headline_l), batch_size=batch_size, shuffle=False)

    # Testing
    test_data_dicts = test_data.collect(engine="streaming").to_dicts()
    test_loader = DataLoader(build_dataset(test_data_dicts, stock2id, e_col, f_cols, t_col, max_headline_l), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train_val_test_split_date(
    data: pl.LazyFrame,
    batch_size: int,
    max_headline_l: int,
    train_start_date: datetime,
    train_end_date: datetime,
    test_start_date: datetime,
    test_end_date: datetime,
    val_ratio=0.1
):
    # Filter by explicit date ranges
    train_data = data.filter((pl.col("Date") >= train_start_date) & (pl.col("Date") <= train_end_date))
    test_data = data.filter((pl.col("Date") >= test_start_date) & (pl.col("Date") <= test_end_date))

    # Training-Validation
    train_data_full = train_data.collect(engine="streaming").to_dicts()
    split_idx = int(len(train_data_full) * (1 - val_ratio))
    train_data_final = train_data_full[:split_idx]
    val_data = train_data_full[split_idx:]

    e_col = "embedded_headline"
    f_cols = ["open", "high"]
    t_col = "close"

    train_loader = DataLoader(build_dataset(train_data_final, stock2id, e_col, f_cols, t_col, max_headline_l), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(build_dataset(val_data, stock2id, e_col, f_cols, t_col, max_headline_l), batch_size=batch_size, shuffle=False)

    # Testing
    test_data_dicts = test_data.collect(engine="streaming").to_dicts()
    test_loader = DataLoader(build_dataset(test_data_dicts, stock2id, e_col, f_cols, t_col, max_headline_l), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Split into train and test
batch_size = 16
train_loader, val_loader, test_loader = train_val_test_split_ratio(data, batch_size, max_headline_l=l.max_headline_len, train_ratio=0.8, val_ratio=0.1)

# start_train = datetime(2018, 1, 1)
# end_train = datetime(2018, 1, 15)
# start_test = datetime(2018, 1, 15)
# end_test = datetime(2018, 1, 30)
# train_loader, val_loader, test_loader = train_val_test_split_date(data, batch_size, train_start_date=start_train, train_end_date=end_train, test_start_date=start_test, test_end_date=end_test, val_ratio=0.1)

criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore padding (=0) tokens
optimizer = optim.AdamW(lstm.parameters(), lr=0.001, weight_decay=1e-5) # AdamW add L2 regularization to punish large weights and prevent overfitting


best_val_loss = float("inf")
patience = 2
counter = 0

# TODO: Steps:
# 1. Models: (Baselines: Only XGBoost/MGNN, addition: LSTM/LLM)
# 2. Make a good, train-test split
# 3. Update the training loop (add model layers, make compatible with diff #features)

print("Training-Val")
max_epochs = 100
for epoch in range(max_epochs):
    lstm.train()
    train_loss  = 0

    progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=True)
    for X_batch, Y_batch, stock_batch, _ in progress_bar:
        X_batch = X_batch.to(device) # shape: (batch_size, seq_len)
        Y_batch = Y_batch.to(device)
        stock_batch = stock_batch.to(device) # shape: (batch_size)
        optimizer.zero_grad()

        # Forward pass
        logits = lstm(X_batch, stock_batch) # shape: (batch_size, vocab_size)

        # # Expand logits to match target sequence length
        logits = logits.unsqueeze(1).repeat(1, Y_batch.size(1), 1) # shape: (batch_size, seq_len, vocab_size)

        flat_logits = logits.view(-1, logits.size(-1)) # shape: (batch_size * seq_len, vocab_size)
        flat_Y = Y_batch.view(-1) # shape: (batch_size * seq_len)
        print("Shapes")
        print(logits.shape)
        print(Y_batch.shape)
        loss = criterion(logits, Y_batch)


        loss.backward()
        optimizer.step()
        train_loss  += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    # Validation
    lstm.eval()
    val_loss = 0
    with torch.no_grad(): # freeze weights
        for X_val, Y_val, stock_val, _ in val_loader:
            X_val, Y_val, stock_val = X_val.to(device), Y_val.to(device), stock_val.to(device)
            logits = lstm(X_val, stock_val)
            # logits = logits.unsqueeze(1).repeat(1, Y_val.size(1), 1)
            val_loss += criterion(logits.view(-1, logits.size(-1)), Y_val.view(-1)).item()

    # Early stopping if val doesn't improve for #patience times
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(lstm.state_dict(), "best_model.pt")
    else:
        counter += 1
        if counter == patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val Loss={val_loss/len(val_loader):.4f}")

lstm.load_state_dict(torch.load("best_model.pt")) # load best model

# region Testing

print("Testing")

batch = next(iter(test_loader), None)
assert batch is not None and len(batch) > 0, "test_data is empty. Probably n was too small, or test %"

h_row_idxs = []
h_ts = []
lstm.eval()
with torch.no_grad():
    for X_test, Y_test, stock_test, row_idx in test_loader:
        X_test, stock_test, row_idx = X_test.to(device), stock_test.to(device), row_idx.to(device)

        h_att = lstm.forward_for_ht(X_test, stock_test)
        h_row_idxs.append(row_idx.cpu())
        h_ts.append(h_att.cpu())

h_idx_tensor = torch.cat(h_row_idxs, dim=0).numpy()
h_ts_tensor = torch.cat(h_ts, dim=0).numpy() # shape: (num_rows, hidden_dim)

lf_h_att = pl.LazyFrame({
    "row_index": h_idx_tensor,
    "lstm_embed": list(h_ts_tensor) # store each row as a list
})

lf_joined = l.lf.join(lf_h_att, on="row_index", how="inner")


start_date = lf_joined.select(pl.min("Date")).collect().item().date()
end_date = lf_joined.select(pl.max("Date")).collect().item().date()

print("Writing to .parquet")
parquet_path = f"lstm_embed_test_{start_date}_{end_date}.parquet"
lf_joined.sink_parquet(parquet_path)
print(f"Saved to {parquet_path}")

print("Loading witten file")
lf_loaded = pl.scan_parquet(parquet_path)
print(lf_loaded.sort("Date").collect())