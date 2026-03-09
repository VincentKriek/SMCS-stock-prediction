from huggingface_hub import hf_hub_download
import polars as pl


def load_hf_lazyframe(repo_id, subfolder, filename, min_date, max_date):
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        repo_type="dataset",
    )

    return (
        pl.scan_csv(
            path,
            ignore_errors=True,
            infer_schema_length=10000,
        )
        .with_columns(
            pl.col("Date")
            .str.replace(" UTC", "")
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
        )
        .filter(pl.col("Date").is_between(min_date, max_date))
    )
