import polars as pl


def aggregate_scores(news_lf, stock_lf, lambda_decay):
    news_lf = group_lf(news_lf)
    news_lf = aggregate_scores_mean(news_lf)
    news_lf = aggregate_scores_median(news_lf)
    news_lf = aggregate_scores_mode(news_lf)

    print(stock_lf.head(10).collect())

    joined_lf = stock_lf.join(news_lf, on="Date", how="left")

    final_lf = joined_lf

    return final_lf


def fill_news_gaps(lf, lambda_decay):
    pass


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
    print(aggregate_scores(news_lf, stock_lf, 0.03).collect())
