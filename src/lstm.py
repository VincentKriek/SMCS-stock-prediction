import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import string
import tqdm

print(torch.__version__)
print("CUDA:", torch.cuda.is_available())

nltk.download('punkt_tab')
nltk.download('stopwords')


class LazyHeadlineVectorizer:
    def __init__(self, parquet_path, col_name="Article_title", vector_size=128, window=5, min_count=1):
        self.parquet_path = parquet_path
        self.col_name = col_name
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count

        self.stop_words = set(stopwords.words("english"))
        self.punctuation = set(string.punctuation)

        self.num_rows = None
        self.lf = None
        self.model = None
        self.max_headline_len = None
        self.word2id = {}

    def load_headlines(self, n=None):
        self.lf = pl.scan_parquet(self.parquet_path)
        if n:
            self.lf = self.lf.head(n)
        self.num_rows = self.lf.select(pl.len()).collect().item()
        return self.lf

    # Tokenizer for a single headline
    def clean_tokenize(self, headline):
        tokens = word_tokenize(headline.lower())
        tokens = [t for t in tokens if t not in self.stop_words
                  and t not in self.punctuation
                  and not t.isnumeric()]
        return tokens
    
    def tokenize_lf(self):
        self.lf = self.lf.with_columns(
                tokenized_headline=pl.col(self.col_name).map_elements(
                l.clean_tokenize,
                return_dtype=pl.List(pl.String)
            )
        ).with_columns(
            headline_len=pl.col("tokenized_headline").list.len()
        )

        # Set the max allowed healdine length for lstm to the 95th percentile
        self.max_headline_len = int((
            self.lf
            .select("headline_len")
            .quantile(0.95, interpolation="linear")
            .collect()
        ).item())

    # Iterator for Word2Vec
    class HeadlinesIterator:
        def __init__(self, lf):
            self.lf: pl.LazyFrame = lf

        def __iter__(self):
            for row in self.lf.select("tokenized_headline").collect(engine="streaming").iter_rows():
                yield row[0]

    # Train Word2Vec lazily
    def train_word2vec(self):
        sentences = self.HeadlinesIterator(self.lf)
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=3
        )
        return self.model

    def build_vocab_id(self):
        self.word2id = {w: i + 1 for i, w in enumerate(self.model.wv.index_to_key)}
        self.word2id["<PAD>"] = 0

    def add_embedded_column(self):
        def process(tokens):
            ids = [self.word2id.get(t, 0) for t in tokens] # convert token to int

            # LSTM needs equal length input sequences, add padding to equal sentence lengths
            if len(ids) < self.max_headline_len:
                ids = ids + [0] * (self.max_headline_len - len(ids))
            else:
                ids = ids[:self.max_headline_len]

            return ids

        self.lf = self.lf.with_columns(
            embedded_headline=pl.col("tokenized_headline").map_elements(
                process,
                return_dtype=pl.List(pl.Int64)
            )
        )

        return self.lf
    
    # Map the Word2Vec embedding to each token in the vocab
    def build_embedding_matrix(self):
        vocab_size = len(self.word2id)
        emb_dim = self.vector_size

        matrix = np.zeros((vocab_size, emb_dim))

        for word, idx in self.word2id.items():
            if word in self.model.wv:
                matrix[idx] = self.model.wv[word]

        self.embedding_matrix = matrix
    
    def run(self, n=None):
        self.load_headlines(n)
        print("")
        self.tokenize_lf()
        self.train_word2vec()
        self.build_vocab_id()
        self.add_embedded_column()
        self.build_embedding_matrix()

    def run(self, n=None):
        print("Loading headlines...")
        self.load_headlines(n)
        print("Tokenizing...")
        self.tokenize_lf()
        print("Training Word2Vec model...")
        self.train_word2vec()
        print("Building vocabulary IDs...")
        self.build_vocab_id()
        print("Adding embedded column...")
        self.add_embedded_column()
        print("Building embedding matrix...")
        self.build_embedding_matrix()
        print("run() finished")

l = LazyHeadlineVectorizer("../news_formatted_2018-01-01_2023-12-31.parquet")
l.run(n=50)
# print(l.lf.select(pl.len()).collect()) # #rows


# ===== Attention Mechanism =====
# Based on code from: https://www.ijcai.org/proceedings/2020/0626.pdf
class Attentive_Pooling(nn.Module):
    def __init__(self, hidden_dim):
        super(Attentive_Pooling, self).__init__()
        self.w_1 = nn.Linear(hidden_dim, hidden_dim) # Matrix for memory
        self.w_2 = nn.Linear(hidden_dim, hidden_dim) # Matrix for query
        self.u = nn.Linear(hidden_dim, 1, bias=False) # scores how imporant h is

    # LINEAR ADDITIVE ATTENTION MECHANISM EQ. (2) AND (3)
    def forward(self, memory, query=None, mask=None):
        '''
        :param query:  (node, hidden)
        :param memory: (node, hidden)
        :param mask:
        :return:
        '''
        # print("IN ATT shapes:")
        # print(memory.shape)
        # print(query.shape)

        if query is None:
            h = torch.tanh(self.w_1(memory))  # shape: (node, hidden)
        else:
            query = query.unsqueeze(1)
            h = torch.tanh(self.w_1(memory) + self.w_2(query))  # shape: (node, hidden)

        score = torch.squeeze(self.u(h), -1)  # score = u * h
        if mask is not None:
            score = score.masked_fill(mask.eq(0), -1e9) # set score = -infinity on masked places
        alpha = F.softmax(score, -1)  # shape: (node)
        s = torch.sum(torch.unsqueeze(alpha, -1) * memory, -2) # Eq. (3)
        return s # shape: (hidden_dim, 1)

# ===== LSTM Encoder =====
class LSTM_Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, embedding_matrix, num_stocks, stock_emb_dim, layer_dim=1):
        super(LSTM_Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.tensor(embedding_matrix)) # Copy the Word2Vec embeddings
        self.stock_embedding = nn.Embedding(num_stocks, stock_emb_dim) # Learn the stock_emb from the data itself

        # hidden_dim = dim of h_t
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.att = Attentive_Pooling(hidden_dim)

        # Fully connected Layer from hidden_dim that will give the 'probs' for each possible next word
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, stock_ids, h_in=None, mem_in=None):
        # x.shape = (batch_size, seq_len)
        # x_emb.shape = (batch_size, seq_len, embed_dim)
        x_embedding = self.embedding(x) # lookup the embedding

        if h_in is None or mem_in is None:
            h_in = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            mem_in = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)

        out, (h_out, c_out) = self.lstm(x_embedding, (h_in, mem_in))
        # out.shape: (batch_size, seq_len, hidden_dim)
        # h_out.shape: (batch_size, 1, hidden_dim)

        # Apply attention mechanism
        stock_emb = self.stock_embedding(stock_ids) # (batch_size, stock_dim)
        mask = (x != 0) # Mask to give no weight to the pad tokens in the attention
        h_att = self.att(out, query=stock_emb, mask=mask)

        logits = self.fc(h_att)
        return logits # shape: (batch_size, vocab_size)
    
    def forward_for_ht(self, x: torch.Tensor, stock_ids, h_in=None, mem_in=None):
        x_embedding = self.embedding(x) # lookup the embedding
        if h_in is None or mem_in is None:
            h_in = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            mem_in = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)

        out, (h_out, c_out) = self.lstm(x_embedding, (h_in, mem_in))

        # Apply attention mechanism
        stock_emb = self.stock_embedding(stock_ids) # (batch_size, stock_dim)
        mask = (x != 0) # Mask to give no weight to the pad tokens in the attention
        h_att = self.att(out, query=stock_emb, mask=mask)
        return h_att

num_stocks = l.lf.select(pl.col("Stock_symbol").n_unique()).collect().item()
lstm = LSTM_Encoder(
    vocab_size=len(l.word2id),
    embedding_dim=l.vector_size,
    hidden_dim=128,
    embedding_matrix=l.embedding_matrix,
    num_stocks=num_stocks,
    stock_emb_dim=128,
    layer_dim = 1
)

# TODO: Next steps:
# 1. Add attention mechanism on top of the lstm (Done??)
# 2. Train the LSTM on the 'train headlines'
# 3. Somehow store a output vector h_t in a Lazyframe

# region Training

def build_dataset(data, stock2id: dict[str, int]):
    X_list, Y_list, stock_ids_list, row_idx_list = [], [], [], []

    for row in data:
        seq = row["embedded_headline"]
        X_list.append(seq[:-1]) # all words except last
        Y_list.append(seq[1:]) # next word as target (a 'shift' by one per word)
        stock_ids_list.append(row["Stock_symbol"])
        row_idx_list.append(row["row_index"]) # store original row idx

    X = torch.tensor(X_list, dtype=torch.long)
    Y = torch.tensor(Y_list, dtype=torch.long)
    stock_ids = torch.tensor([stock2id[s] for s in stock_ids_list], dtype=torch.long)
    row_idx_tensor = torch.tensor(row_idx_list, dtype=torch.long)

    return TensorDataset(X, Y, stock_ids, row_idx_tensor)

# Map stock symbols (string) to ints:
print("="*40)
print("Loading Data")
word2id = {word: idx for word, idx in l.word2id.items()}
id2word = {idx: word for word, idx in l.word2id.items()}

stocks = sorted(l.lf.select("Stock_symbol").unique().collect().to_series().to_list())
stock2id = {symbol: idx for idx, symbol in enumerate(stocks)}
id2stock = {idx: symbol for idx, symbol in enumerate(stocks)}

# Collect embedded headlines from LazyFrame
data = l.lf.select("row_index", "embedded_headline", "Stock_symbol", "Date")
data = data.sort("Date")

# Split into train and test
train_ratio = 0.8
split_date = data.select("Date").collect()[int(train_ratio * l.num_rows)].item()
train_data = data.filter(pl.col("Date") <= split_date)
test_data = data.filter(pl.col("Date") > split_date)

dataset = build_dataset(train_data.collect(engine="streaming").to_dicts(), stock2id)
batch_size = 2 # wrap in DataLoader for batching
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ===== Training =====
train_data_full = train_data.collect(engine="streaming").to_dicts()
val_ratio = 0.1
split_idx = int(len(train_data_full) * (1 - val_ratio))
train_data_final = train_data_full[:split_idx]
val_data = train_data_full[split_idx:]

train_dataset = build_dataset(train_data_final, stock2id)
val_dataset = build_dataset(val_data, stock2id)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore padding (=0) tokens
optimizer = optim.Adam(lstm.parameters(), lr=0.001)

print("Training-Val")
epochs = 5
for epoch in range(epochs):
    lstm.train()
    train_loss  = 0

    progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
    for X_batch, Y_batch, stock_batch, _ in progress_bar:
        X_batch = X_batch.to(device) # shape: (batch_size, seq_len)
        Y_batch = Y_batch.to(device)
        stock_batch = stock_batch.to(device) # shape: (batch_size)
        optimizer.zero_grad()

        # Forward pass
        logits = lstm(X_batch, stock_batch) # shape: (batch_size, vocab_size)
        # Expand logits to match target sequence length
        logits = logits.unsqueeze(1).repeat(1, Y_batch.size(1), 1) # shape: (batch_size, seq_len, vocab_size)

        flat_logits = logits.view(-1, logits.size(-1)) # shape: (batch_size * seq_len, vocab_size)
        flat_Y = Y_batch.view(-1) # shape: (batch_size * seq_len)
        loss = criterion(flat_logits, flat_Y)

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
            logits = logits.unsqueeze(1).repeat(1, Y_val.size(1), 1)
            val_loss += criterion(logits.view(-1, logits.size(-1)), Y_val.view(-1)).item()

    print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val Loss={val_loss/len(val_loader):.4f}")

    # # For testing, look how the prediction of the next word changes over iters
    # lstm.eval()
    # with torch.no_grad():
    #     stock_id = dataset.tensors[2][0].unsqueeze(0).to(device)
    #     X = dataset.tensors[0][0].unsqueeze(0).to(device)
    #     logits = lstm(X, stock_id)
    #     probs = torch.softmax(logits, dim=-1)
    #     pred_word_idx = torch.argmax(probs, dim=-1)
    #     print("Pred next word for: ")
    #     print(id2stock[stock_id[0].item()])
    #     print(id2word[X[0][0].item()], end=" => ")
    #     print(id2word[pred_word_idx.item()])

# region Testing

print("Testing")
test_data_dicts = test_data.collect(engine="streaming").to_dicts()
test_dataset = build_dataset(test_data_dicts, stock2id)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

assert len(test_data_dicts) > 0, "test_data is empty. Probably n was too small, or test %"

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

# print("="*60)
# print(lf_joined.sort("Date").collect())
# print(l.lf.sort("Date").collect())

print("Writing to .parquet")
parquet_path = "test_with_lstm.parquet"
lf_joined.sink_parquet(parquet_path)
print(f"Saved to {parquet_path}")

# print("Loading witten file")
# lf_loaded = pl.scan_parquet(parquet_path)
# print(lf_loaded.collect())