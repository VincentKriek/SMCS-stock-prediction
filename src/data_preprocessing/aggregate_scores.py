import polars as pl


def aggregate_scores(news_lf, stock_lf, lambda_decay):
    news_lf = group_lf(news_lf)
    news_lf = aggregate_scores_mean(news_lf)
    news_lf = aggregate_scores_median(news_lf)
    news_lf = aggregate_scores_mode(news_lf)

    joined_lf = stock_lf.join(news_lf, on=["Date", "Stock_symbol"], how="left")

    joined_lf = fill_news_gaps(joined_lf, lambda_decay, "Sentiment_llm_mean")
    joined_lf = fill_news_gaps(joined_lf, lambda_decay, "Sentiment_llm_median")
    joined_lf = fill_news_gaps(joined_lf, lambda_decay, "Sentiment_llm_mode")

    return joined_lf.drop(
        [
            "Sentiment_llm",
        ]
    )


def fill_news_gaps(lf, lambda_decay, sentiment_col):
    lf = lf.sort(["Stock_symbol", "Date"])

    # add columns for propagation formula
    lf = lf.with_columns(
        pl.col(sentiment_col).forward_fill().over("Stock_symbol").alias("S0"),
        pl.when(pl.col(sentiment_col).is_not_null())
        .then(pl.col("Date"))
        .otherwise(None)
        .forward_fill()
        .over("Stock_symbol")
        .alias("last_news_date"),
    )
    lf = lf.with_columns(
        (pl.col("Date") - pl.col("last_news_date")).dt.total_days().alias("t")
    )

    # apply formula
    lf = lf.with_columns(
        (3 + (pl.col("S0") - 3) * (-lambda_decay * pl.col("t")).exp()).alias(
            f"{sentiment_col}_filled"
        )
    )

    return lf.drop(["S0", "last_news_date", "t", sentiment_col])


def aggregate_scores_mean(lf):
    return lf.with_columns(
        pl.col("Sentiment_llm").list.mean().alias("Sentiment_llm_mean")
    )


def aggregate_scores_median(lf):
    return lf.with_columns(
        pl.col("Sentiment_llm").list.median().alias("Sentiment_llm_median")
    )


def aggregate_scores_mode(lf):
    return lf.with_columns(
        pl.col("Sentiment_llm")
        .list.eval(pl.element().mode())
        .list.first()
        .cast(pl.Float64)
        .alias("Sentiment_llm_mode")
    )


def group_lf(lf):
    return lf.group_by(["Date", "Stock_symbol"]).agg(pl.all())


if __name__ == "__main__":
    news_lf = pl.scan_parquet(
        "data/pre-processor/news_scored_2018-01-01_2023-12-31-batch1.parquet"
    )
    stock_lf = pl.scan_parquet(
        "data/loader/prices_loaded_2018-01-01_2023-12-31.parquet"
    )
    aggregate_scores(news_lf, stock_lf, 0.03).sink_parquet(
        "data/pre-processor/prepared_data_2018-01-01_2023-12-31.parquet"
    )
