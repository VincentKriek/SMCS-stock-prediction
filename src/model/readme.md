# Model

## Preparation

In the project root create the `data/graphs`, `data/model/graphs`, `data/model/output` and `data/pre-processor` directories.

```sh
data
├── graphs
├── pre-processor
└── model
    ├── graphs
    └── output
```

## Running the code

### Running the baselines
To run the baseline models run:
```sh
uv run src\model\baseline.py
```
At the top of ` src\model\baseline.py` there is a variable `USE_LIN_REGR`.
- Set it to `True` to use **linear regression**.
- Set it to `False` to use **MLP**.

The code should output a csv file with the evaluation metrics in `data/model/output` with the name:
```sh
BASE_global_{model_name}_metrics.csv
```

### Running the proposed model
### Running the proposed model

1. First, to create the adjacency matrices for the graph, run:

   ```sh
   uv run src\model\create_adj_matrices.py
   ```

   This script processes raw SEC 13F data together with stock metadata to construct the relational structure of the market. It takes as input:

   * preprocessed stock data (`data/pre-processor/prepared_data_2018-01-01_2023-12-31.parquet`),
   * CUSIP-to-stock-symbol mapping (`data/graphs/input_data/CUSIP.csv` as a backup),
   * SEC 13F filings (`SUBMISSION.tsv`, `INFOTABLE.tsv`, `COVERPAGE.tsv`, and FTD files).

   The script performs:

   * mapping between CUSIP identifiers and stock symbols over time mainly using FTD data,
   * extraction of active institutional holdings,
   * aggregation of node and edge features.

   It outputs the following**parquet files**

   1. nodes_stock.parquet
   2. nodes_bank.parquet
   3. edges_bank_stock.parquet
   4. edges_stock_stock.parquet

   in:

   ```sh
   data/graphs/
   ```

   containing:

   1. stock node features,
   2. bank node features,
   3. bank–stock edge data,
   4. stock–stock edge data.

---

2. Next to build the relational graph run:

   ```sh
   uv run src\model\build_graphs.py
   ```

   This script reads the parquet files generated in the previous step and constructs **quarterly graph snapshots**. For each time period, it creates:

   * node feature tensors for stocks and banks,
   * edge indices and edge attributes for all relation types.

   The graph snapshots are saved as PyTorch files (`.pt`) in:

   ```sh
   data/model/graphs/
   ```

   with filenames such as:

   ```sh
   graphs_split_1.pt, graphs_split_2.pt, ...
   ```

---

3. Then to run the proposed model itself (training, validation and testing), run:

   ```sh
   uv run src\model\model.py
   ```

   This script:

   * loads the preprocessed dataset (including numerical features and news data),
   * loads graph snapshots (if MDGNN is enabled),
   * constructs the selected model configuration,
   * performs training using a rolling-window setup (6 months train, 1 month validation, 5 months test),
   * and generates predictions for each split.

   At the top of `src\model\model.py`, the following variables control the experiment:

   * `USE_LSTM`: enables LSTM-based headline embeddings,
   * `USE_MDGNN`: enables graph-based modeling,
   * `LLM_SENTIMENT_MODE`: selects sentiment aggregation ("mean", "median", "mode", or `None`).

   These settings allow running different ablation models:

   * MDGNN only,
   * MDGNN + LSTM,
   * MDGNN + LLM,
   * MDGNN + LSTM + LLM.

   The script outputs prediction files in:

   ```sh
   data/model/output/
   ```

   with filenames such as:

   ```sh
   preds_{experiment_name}_split_{i}.csv
   ```

   where `{experiment_name}` reflects the selected configuration (e.g., `lstm_mdgnn_llm_mean`).


### Analysis
To calculate the evaluation metrics for the proposed model (or any ablation) run:
```sh
uv run src\model\analyse.py
```

The code should output a csv file with the evaluation metrics in `data/model/output` with the name:
```sh
{model_name}_metrics.csv
```