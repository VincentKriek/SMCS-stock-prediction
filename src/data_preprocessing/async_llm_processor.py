import asyncio
import os
import json
import pyarrow.parquet as pq
import polars as pl
from pathlib import Path
from dotenv import load_dotenv
from ollama_functions import score_batch
from tqdm import tqdm


load_dotenv()

WORKERS = 2  # Can increase concurrency
START_DATE = os.environ["MIN_DATE"]
END_DATE = os.environ["MAX_DATE"]
INPUT_FILE = Path(f"data/pre-processor/news_formatted_{START_DATE}_{END_DATE}.parquet")
OUTPUT_FILE = Path(f"data/pre-processor/news_scored_{START_DATE}_{END_DATE}.parquet")
CHECKPOINT_FILE = Path(f"data/pre-processor/checkpoint_{START_DATE}_{END_DATE}.jsonl")


# -----------------------------
# Checkpoint writer
# -----------------------------
async def checkpoint_writer(save_queue, pbar):  # Added pbar here
    """Listen for completed rows and append them immediately."""
    with open(CHECKPOINT_FILE, "a") as f:
        while True:
            item = await save_queue.get()
            if item is None:
                break

            row_index, score = item
            if score is not None:
                f.write(
                    json.dumps({"row_index": row_index, "Sentiment_llm": score}) + "\n"
                )
                f.flush()

            pbar.update(1)  # Update the bar here!
            save_queue.task_done()


# -----------------------------
# Worker function
# -----------------------------
async def give_scores(task_queue, save_queue):
    while True:
        item = await task_queue.get()
        if item is None:
            task_queue.task_done()
            break

        row_index, row = item

        # Score single row
        score = await score_batch([row])  # still returns a list of one score
        await save_queue.put((row_index, score[0] if score else None))

        task_queue.task_done()


# -----------------------------
# Load checkpoints
# -----------------------------
def load_checkpoints():
    processed_indices = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_indices.add(data["row_index"])
                except json.JSONDecodeError:
                    continue
    return processed_indices


# -----------------------------
# Stream rows
# -----------------------------
def stream_rows(processed_indices):
    parquet_file = pq.ParquetFile(INPUT_FILE)

    for batch in parquet_file.iter_batches(
        columns=["Article_title", "Stock_symbol", "summary", "row_index"]
    ):
        df = pl.from_arrow(batch)
        # Filter out already processed rows
        df = df.filter(~pl.col("row_index").is_in(processed_indices))

        # Yield each row individually from the batch
        for row in df.to_dicts():
            yield row["row_index"], row


# -----------------------------
# Orchestrator
# -----------------------------
async def process():
    processed_indices = load_checkpoints()
    print(f"Loaded {len(processed_indices)} previously processed rows.")

    df = pl.scan_parquet(INPUT_FILE)
    if processed_indices:
        df = df.filter(~pl.col("row_index").is_in(processed_indices))

    total_rows = df.select(pl.count()).collect()[0, 0]

    task_queue = asyncio.Queue()
    save_queue = asyncio.Queue()

    # Define the progress bar here
    with tqdm(total=total_rows, desc="Scoring rows") as pbar:

        # Pass pbar to the writer
        workers = [
            asyncio.create_task(give_scores(task_queue, save_queue))
            for _ in range(WORKERS)
        ]
        writer_task = asyncio.create_task(checkpoint_writer(save_queue, pbar))

        # Feed the queue (No pbar.update here!)
        for row_index, row in stream_rows(processed_indices):
            await task_queue.put((row_index, row))

        # Wait for workers to finish the work
        await task_queue.join()

        # Stop workers and writer as before
        for _ in workers:
            await task_queue.put(None)
        await asyncio.gather(*workers)

        await save_queue.put(None)
        await writer_task


# -----------------------------
# Write final parquet
# -----------------------------
def write_final_results():
    if not os.path.exists(CHECKPOINT_FILE):
        print("No checkpoints found. Nothing to write.")
        return

    results_df = pl.read_ndjson(CHECKPOINT_FILE)
    lf = pl.scan_parquet(INPUT_FILE)
    (lf.join(results_df.lazy(), on="row_index", how="left").sink_parquet(OUTPUT_FILE))
    print(f"Successfully saved final results to {OUTPUT_FILE}")


async def generate_sentiment_scores():
    await process()
    write_final_results()


if __name__ == "__main__":
    asyncio.run(generate_sentiment_scores())
