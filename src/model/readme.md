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
1. First, to create the adjacency matrices for the graph, run:
    ```sh
    uv run src\model\create_adj_matrices.py
    ```
    This ..., and should output ... (a parquet file?, with the name(s): ...)
2. Next to build the relational graph run:
    ```sh
    uv run src\model\build_graphs.py
    ```
    This ..., and should output ...
3. Then to run the proposed model itself (training, validation and testing), run:
    ```sh
    uv run src\model\model.py
    ```
    At the top of ` src\model\model.py` there are three variables: `USE_LSTM`, `USE_MDGNN`, and `LLM_SENTIMENT_MODE`. These can be set to different values to select different ablation models.

    This ..., and should output ...

### Analysis
To calculate the evaluation metrics for the proposed model (or any ablation) run:
```sh
uv run src\model\analyse.py
```

The code should output a csv file with the evaluation metrics in `data/model/output` with the name:
```sh
{model_name}_metrics.csv
```