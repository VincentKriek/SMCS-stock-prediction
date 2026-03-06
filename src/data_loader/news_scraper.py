import polars as pl
import aiohttp
import asyncio
import trafilatura


async def fetch_article(session, url):
    try:
        async with session.get(url) as resp:
            html = await resp.text()
            return trafilatura.extract(html)
    except Exception:
        return None


async def fetch_batch(urls):
    async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as session:
        tasks = [fetch_article(session, u) for u in urls]
        return await asyncio.gather(*tasks)


def add_article_column_stream(lf: pl.LazyFrame, url_column="Url", batch_size=100):

    result_frames = []
    lf = lf.drop("Article")

    for batch in lf.select(url_column).collect(streaming=True).iter_slices(batch_size):
        urls = batch[url_column].to_list()

        articles = asyncio.run(fetch_batch(urls))

        result_frames.append(pl.DataFrame({url_column: urls, "Article": articles}))

    article_df = pl.concat(result_frames)

    return lf.join(article_df.lazy(), on=url_column)
