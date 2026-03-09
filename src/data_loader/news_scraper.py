import polars as pl
import asyncio
import trafilatura
from typing import List
from concurrent.futures import ProcessPoolExecutor

# Controls max network concurrency per async loop
CONCURRENCY_LIMIT = 50

# Number of parallel processes for CPU-bound extraction
PROCESS_CHUNKS = 4


# --------------------------
# Async fetch wrapper
# --------------------------
async def fetch_article(url: str, semaphore: asyncio.Semaphore):
    if not url:
        return None
    async with semaphore:
        try:
            result = await trafilatura.async_fetch_url(
                url, user_agent="Mozilla/5.0", timeout=15
            )
            return result
        except Exception:
            return None


async def download_chunk(urls: List[str]):
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [fetch_article(url, semaphore) for url in urls]
    return await asyncio.gather(*tasks)


def run_async_fetch(urls: List[str]) -> List[str]:
    """Run a chunk of URLs in an asyncio loop"""
    return asyncio.run(download_chunk(urls))


# --------------------------
# Main Polars integration
# --------------------------
def add_article_column_stream(lf: pl.LazyFrame, url_column="Url") -> pl.LazyFrame:
    """
    Efficiently fetch articles using async + multiple processes.
    """
    # 1️⃣ Get unique URLs
    df_urls = lf.select(url_column).unique().collect()
    urls = df_urls[url_column].to_list()
    if not urls:
        return lf.with_columns(pl.lit(None).alias("Article"))

    # 2️⃣ Split URLs into chunks for parallel processes
    chunk_size = max(1, len(urls) // PROCESS_CHUNKS)
    url_chunks = [urls[i : i + chunk_size] for i in range(0, len(urls), chunk_size)]

    # 3️⃣ Run parallel processes
    with ProcessPoolExecutor(max_workers=PROCESS_CHUNKS) as executor:
        results_chunks = list(executor.map(run_async_fetch, url_chunks))

    # 4️⃣ Flatten results
    articles = [item for sublist in results_chunks for item in sublist]

    # 5️⃣ Join back to the LazyFrame
    mapping_df = pl.DataFrame({url_column: urls, "Article": articles})
    return lf.join(mapping_df.lazy(), on=url_column, how="left")
