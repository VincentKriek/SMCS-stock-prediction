import torch
import polars as pl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1️⃣ Model ID
model_id = "sshleifer/distilbart-cnn-12-6"

# 2️⃣ Load tokenizer and seq2seq model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, use_safetensors=True)

# 3️⃣ Detect hardware
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"🚀 Using device: {device}")


# 4️⃣ Define batch summarizer
def distilbart_summarizer_batch(batch_series: pl.Series) -> pl.Series:
    texts = batch_series.fill_null("").to_list()

    # Skip very short texts
    valid_indices = [i for i, t in enumerate(texts) if len(t) > 50]
    if not valid_indices:
        return pl.Series([None] * len(texts))

    summaries = [None] * len(texts)

    # Process in a simple for-loop batch-wise
    for i in range(0, len(valid_indices), 8):
        chunk_idxs = valid_indices[i : i + 8]
        chunk_texts = [texts[idx] for idx in chunk_idxs]

        # Tokenize batch
        inputs = tokenizer(
            chunk_texts,
            max_length=1024,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        ).to(device)

        # Generate summaries
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=130,
            min_length=30,
            num_beams=4,
            length_penalty=2.0,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

        # Decode and assign
        decoded = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        for idx, text_summary in zip(chunk_idxs, decoded):
            summaries[idx] = text_summary

    return pl.Series(summaries)


# 5️⃣ Polars helper to add summary column
def add_summary_column(lf: pl.LazyFrame, text_column: str = "Article") -> pl.LazyFrame:
    return lf.with_columns(
        pl.col(text_column)
        .map_batches(distilbart_summarizer_batch, return_dtype=pl.String)
        .alias("DistilBART_summary")
    )
