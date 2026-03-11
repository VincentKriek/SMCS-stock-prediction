import polars as pl
from concurrent.futures import ProcessPoolExecutor
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# Global instances (initialized per process)
_summarizer = TextRankSummarizer()
_tokenizer = Tokenizer("english")


def single_article_summarizer(text: str) -> str:
    """Core logic for one article."""
    if not text or len(text) < 150:
        return text

    try:
        parser = PlaintextParser.from_string(text, _tokenizer)
        summary = _summarizer(parser.document, 3)
        return " ".join([str(s) for s in summary])
    except Exception as e:
        print(e)
        return text


def add_summary_column(lf: pl.LazyFrame) -> pl.LazyFrame:
    df = lf.filter(
        pl.col("Article").is_not_null()
    ).collect()  # don't include rows without an article body

    # 2. Use Multiprocessing to process the 'Article' column
    # We use a ProcessPool to bypass the Python GIL
    articles = df["Article"].to_list()

    with ProcessPoolExecutor() as executor:
        # map() handles the distribution across your CPU cores
        summaries = list(executor.map(single_article_summarizer, articles))

    # 3. Add the results back and return as LazyFrame
    return df.with_columns(pl.Series("summary", summaries, dtype=pl.String)).lazy()
