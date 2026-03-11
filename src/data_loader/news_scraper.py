# this code has been generated with the help of GenAI

import asyncio
from concurrent.futures import ProcessPoolExecutor
import httpx
import trafilatura
import polars as pl
from typing import List, Optional

# Constants
CONCURRENCY_LIMIT = 40  # Number of simultaneous downloads
MAX_RETRIES = 2


async def download_html(
    url: str, client: httpx.AsyncClient, semaphore: asyncio.Semaphore
) -> Optional[str]:
    """Purely downloads the raw HTML with retry logic."""
    if not url:
        return None

    async with semaphore:
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await client.get(url, timeout=12, follow_redirects=True)
                if response.status_code == 200:
                    return response.text
                if response.status_code == 404:  # Don't retry broken links
                    return None
            except Exception:
                if attempt == MAX_RETRIES:
                    return None
                await asyncio.sleep(1)  # Small backoff
        return None


def extract_content(html: Optional[str]) -> Optional[str]:
    """CPU-bound task: Extracts text from HTML using trafilatura."""
    if not html:
        return None
    # include_tables is important for financial news (earnings etc)
    return trafilatura.extract(
        html, include_comments=False, include_tables=True, favor_recall=True
    )


async def fetch_all_htmls(urls: List[str]) -> List[Optional[str]]:
    """Handles the async network phase."""
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/119.0.0.0"
    }

    async with httpx.AsyncClient(headers=headers, http2=True) as client:
        tasks = [download_html(url, client, semaphore) for url in urls]
        return await asyncio.gather(*tasks)


def add_article_column_stream(lf: pl.LazyFrame, url_column="Url") -> pl.LazyFrame:
    lf = lf.drop("Article")

    # 1. Get unique URLs
    df_urls = lf.select(url_column).unique().collect()
    urls = df_urls[url_column].to_list()

    # exit if no urls
    if not urls:
        return lf.with_columns(pl.lit(None).alias("Article"))

    # 2. Network Phase: Fetch all HTML (Async)
    raw_htmls = asyncio.run(fetch_all_htmls(urls))

    # 3. CPU Phase: Extract text from HTML (Parallel)
    with ProcessPoolExecutor() as executor:
        articles = list(executor.map(extract_content, raw_htmls))

    # 4. Join back
    mapping_df = pl.DataFrame(
        {url_column: urls, "Article": articles},
        schema={url_column: pl.Utf8, "Article": pl.Utf8},
    )

    return (
        lf.join(mapping_df.lazy(), on=url_column, how="left")
        .with_columns(pl.col("Article"))
        .filter(
            ~pl.col("Article").str.contains(
                "Never miss a trade again with the fastest news alerts in the world!"
            )  # filter out paywall protected articles
        )
    )
