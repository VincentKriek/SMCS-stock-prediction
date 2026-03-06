import polars as pl
import spacy
import pytextrank  # noqa: F401

# 1. Setup spaCy with PyTextRank
# We load the "small" model for speed, as the paper focuses on efficiency
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")


def textrank_summarizer_batch(
    batch_texts: pl.Series, top_n_sentences: int = 3
) -> pl.Series:
    """
    Processes a batch of texts and returns their TextRank summaries.
    """
    summarized_texts = []

    # Process the batch using spacy.pipe for faster multi-threading
    for doc in nlp.pipe(batch_texts.fill_null("").to_list(), n_process=1):
        # The paper selects the top sentences based on their rank
        # We join the top N sentences to form the summary
        summary_sentences = [
            sent.text
            for sent in doc._.textrank.summary(
                limit_phrases=15, limit_sentences=top_n_sentences
            )
        ]

        if not summary_sentences:
            summarized_texts.append("")
        else:
            summarized_texts.append(" ".join(summary_sentences))

    return pl.Series(summarized_texts)


def add_summary_column(lf: pl.LazyFrame, text_column: str = "Article") -> pl.LazyFrame:
    lf = lf.drop("Textrank_summary")
    return lf.with_columns(
        [
            pl.col(text_column)
            .map_batches(lambda s: textrank_summarizer_batch(s, top_n_sentences=3))
            .alias("Textrank_summary")
        ]
    )
