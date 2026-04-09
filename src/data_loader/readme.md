# Data loader

## Preparation

In the project root create the `data/loader`, `data/loader/batches`, `data/loader/batches/external`, `data/loader/batches/nasdaq` and `data/loader/batches/price`.

```sh
data/loader
└── batches
    ├── external
    ├── nasdaq
    └── price
```

## Running the code

This code can load both stock data and news data from the FNSPID huggingface repository. Which data is loaded can be altered by setting the `HF_SUBFOLDER` variable in the .env file. The value of this variable can either be `Stock_news` or `Stock_price`. The code can then be run by running

```sh
$ uv run src/data_loader/main.py
```

from the project root. This code should eventually output a parquet file which can be used in the pre-processing step. The parquet file should have the following name:

```
{{news|prices}}_loaded_{{min_date}}_{{max_date}}.parquet
```
