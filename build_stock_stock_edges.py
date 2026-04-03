"""
MDGNN Stock-Stock Edge Construction (E_SS)

Two stocks are linked if they are co-held by the same institution
in the same quarter. Edge weight = number of institutions that
co-hold both stocks in that quarter.

Input:  graph_output/edges_bank_stock.parquet
Output: graph_output/edges_stock_stock.parquet
"""

import pandas as pd
import gc
from itertools import combinations
from tqdm import tqdm
from collections import Counter  

INPUT_PATH  = "graph_output/edges_bank_stock.parquet"
OUTPUT_PATH = "graph_output/edges_stock_stock.parquet"

# Limit portfolio size to avoid combinatorial explosion
MAX_STOCKS_PER_BANK = 100

# Minimum co-holding threshold
MIN_CO_HOLD = 2


# LOAD

print("Loading bank-stock edges ...")
edges_bs = pd.read_parquet(INPUT_PATH, columns=["bank_id", "stock_id", "year", "quarter"])
print(f"  Loaded {len(edges_bs):,} rows")


# BUILD CO-HOLDING PAIRS PER QUARTER

print("Building stock-stock co-holding pairs ...")

all_pairs = []

quarters = edges_bs[["year", "quarter"]].drop_duplicates().sort_values(["year", "quarter"])

with tqdm(quarters.itertuples(index=False), total=len(quarters),
          desc="Quarters", unit="quarter", ncols=80, colour="green") as qbar:

    for row in qbar:
        y, q = row.year, row.quarter
        qbar.set_description(f"Processing {y} Q{q}")

        # All holdings in this quarter
        quarter_edges = edges_bs[
            (edges_bs["year"] == y) & (edges_bs["quarter"] == q)
        ]

        # Group stocks by bank
        bank_groups = quarter_edges.groupby("bank_id")["stock_id"].apply(list)

        # use Counter instead of dict
        pair_counts = Counter()

        for stocks in bank_groups:

            # Skip small or excessively large portfolios
            if len(stocks) < 2:
                continue

            # Limit large banks
            if len(stocks) > MAX_STOCKS_PER_BANK:
                continue

            # Generate all stock pairs
            pair_counts.update(combinations(sorted(stocks), 2))

        if not pair_counts:
            continue

        # Convert to DataFrame
        pairs_df = pd.DataFrame([
            {"stock_id_1": k[0], "stock_id_2": k[1], "co_holder_count": v}
            for k, v in pair_counts.items()
        ])

        # Filter weak edges
        pairs_df = pairs_df[pairs_df["co_holder_count"] >= MIN_CO_HOLD]

        if len(pairs_df) == 0:
            continue

        pairs_df["year"]    = y
        pairs_df["quarter"] = q
        pairs_df["edge_type"] = "co_held_by_institution"

        all_pairs.append(pairs_df)

        tqdm.write(f"  {y} Q{q}: {len(pairs_df):,} stock-stock pairs")

        del quarter_edges, bank_groups, pair_counts, pairs_df
        gc.collect()


# COMBINE AND SAVE

print("\nCombining all quarters ...")

if len(all_pairs) == 0:
    raise ValueError("No stock-stock edges were generated. Check filtering thresholds.")

edges_ss = pd.concat(all_pairs, ignore_index=True)

del all_pairs
gc.collect()

print(f"  Total stock-stock edges: {len(edges_ss):,}")
print(f"  Unique stock pairs:      {edges_ss[['stock_id_1','stock_id_2']].drop_duplicates().shape[0]:,}")
print(f"  Avg co-holder count:     {edges_ss['co_holder_count'].mean():.1f}")

edges_ss.to_parquet(OUTPUT_PATH, index=False)

print(f"\n  Saved {OUTPUT_PATH}")
print("Done!")