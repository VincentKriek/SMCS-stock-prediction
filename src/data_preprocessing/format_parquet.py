import polars as pl


def format_parquet(lf: pl.LazyFrame):
    lf = add_index_col(lf)
    return lf


def add_index_col(lf: pl.LazyFrame):
    return lf.with_row_index("row_index")
