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
from datetime import datetime

print(torch.__version__)
print(torch.version.cuda)
print("CUDA:", torch.cuda.is_available())

nltk.download('punkt_tab')
nltk.download('stopwords')

class LazyHeadlineVectorizer:
    def __init__(self, parquet_path, col_name="Article_title", vector_size=128, window=5, min_count=1, n_rows=None, start_date=None, end_date=None):
        self.parquet_path = parquet_path
        self.col_name = col_name
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.start_date = start_date
        self.end_date = end_date # will be used inclusive
        self.n_rows = n_rows

        self.stop_words = set(stopwords.words("english"))
        self.punctuation = set(string.punctuation)

        self.lf = None
        self.model = None
        self.max_headline_len = None
        self.word2id = {}

    def load_headlines(self):
        if self.start_date and self.end_date:
            assert self.start_date < self.end_date, "Start date >= end date"
            self.lf = pl.scan_parquet(self.parquet_path)
            self.lf = self.lf.filter(
                (pl.col("Date") >= self.start_date) &
                (pl.col("Date") <= self.end_date)
            )
            self.n_rows = self.lf.select(pl.len()).collect().item()
        elif self.n_rows:
            self.lf = pl.scan_parquet(self.parquet_path, n_rows=self.n_rows)

        assert self.lf is not None, "LazyFrame is None. Please enter valid dates or n_rows"

        print(f"Num Rows: {self.n_rows}")

        return self.lf

    # Tokenizer for a single headline
    def clean_tokenize(self, headline):
        if isinstance(headline, pl.Series):
            token_list = []
            for h in headline:
                tokens = word_tokenize(h.lower())
                tokens = [t for t in tokens if t not in self.stop_words
                        and t not in self.punctuation
                        and not t.isnumeric()]
                token_list.extend(tokens) # concatenate the headlines tokens
            return token_list
    
        # headline is string
        tokens = word_tokenize(headline.lower())
        tokens = [t for t in tokens if t not in self.stop_words
                  and t not in self.punctuation
                  and not t.isnumeric()]
        return tokens
    
    def tokenize_lf(self):
        self.lf = self.lf.with_columns(
                tokenized_headline=pl.col(self.col_name).map_elements(
                self.clean_tokenize,
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
                h = row[0]
                if h is None: # skip null values
                    continue
                yield h

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
    
    def run(self):
        print("Loading headlines...")
        self.load_headlines()
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

# ===== Attention Mechanism =====
# Based on code from: https://www.ijcai.org/proceedings/2020/0626.pdf
class Attentive_Pooling(nn.Module):
    def __init__(self, hidden_dim, device):
        super(Attentive_Pooling, self).__init__()
        self.device = device
        self.w_1 = nn.Linear(hidden_dim, hidden_dim, device=self.device) # Matrix for memory
        self.w_2 = nn.Linear(hidden_dim, hidden_dim, device=self.device) # Matrix for query
        self.u = nn.Linear(hidden_dim, 1, bias=False, device=self.device) # scores how imporant h is

    # LINEAR ADDITIVE ATTENTION MECHANISM EQ. (2) AND (3)
    def forward(self, memory, query=None, mask=None):
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
        return s # shape: (batch_size, hidden_dim
    
# ===== LSTM Encoder =====
class LSTM_Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, embedding_matrix, num_stocks, stock_emb_dim, layer_dim=1, device="cpu"):
        super(LSTM_Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim, device=self.device)
        self.embedding.weight.data.copy_(torch.tensor(embedding_matrix)) # Copy the Word2Vec embeddings
        self.stock_embedding = nn.Embedding(num_stocks, stock_emb_dim, device=self.device) # Learn the stock_emb from the data itself

        # hidden_dim = dim of h_t
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, device=self.device)

        self.att = Attentive_Pooling(hidden_dim, device=self.device)

        # Fully connected Layer from hidden_dim that will give the 'probs' for each possible next word
        self.fc = nn.Linear(hidden_dim, vocab_size, device=self.device)

    # TODO: Remove this one/change it for in the whole model pipeline
    def forward(self, x: torch.Tensor, stock_ids, h_in=None, mem_in=None):
        # x.shape = (batch_size, seq_len)
        # x_emb.shape = (batch_size, seq_len, embed_dim)
        x_embedding = self.embedding(x) # lookup the embedding

        if h_in is None or mem_in is None:
            h_in = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)
            mem_in = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)

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
            h_in = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)
            mem_in = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)

        out, (h_out, c_out) = self.lstm(x_embedding, (h_in, mem_in))

        # Apply attention mechanism
        stock_emb = self.stock_embedding(stock_ids) # (batch_size, stock_dim)
        mask = (x != 0) # Mask to give no weight to the pad tokens in the attention
        h_att = self.att(out, query=stock_emb, mask=mask)
        return h_att

