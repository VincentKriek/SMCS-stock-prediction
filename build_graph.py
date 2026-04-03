"""
MDGNN Graph Construction — SEC 13F Data (US Market)

Builds bank nodes, stock nodes, and bank-stock edges (E_SB)
across multiple quarters (2018 Q1 — 2023 Q4) from SEC EDGAR 13F filings.

Expected folder structure:
  SEC data/
    2018q1_form13f/
      SUBMISSION.tsv
      COVERPAGE.tsv
      INFOTABLE.tsv
    2018q2_form13f/
      ...
    ...
    2023q4_form13f/
      ...

Output files (in graph_output/):
  nodes_bank.parquet
  nodes_stock.parquet
  edges_bank_stock.parquet
  graph_summary.txt
"""

import pandas as pd
import os
import gc
from tqdm import tqdm


# CONFIG

PARENT_DIR = "SEC data"  # folder containing all quarter subfolders
OUTPUT_DIR = "graph_output"
CHUNKSIZE = 200_000

# All quarters to process
YEARS = range(2018, 2024)  # 2018 to 2023 inclusive
QUARTERS = [1, 2, 3, 4]

# Quarter end dates — used to filter PERIODOFREPORT per quarter
QUARTER_PERIODS = {
    1: "31-MAR",
    2: "30-JUN",
    3: "30-SEP",
    4: "31-DEC",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# HELPER — build expected period strings
# e.g. Q1 2023 folder contains filings for 31-MAR-2023
# BUT the Q1 submission file (filed approx. 45 days after quarter end)
# actually contains the PREVIOUS quarter's period too.
# We just load ALL rows from each folder — no period filter needed
# since each folder is already scoped to one quarter's submissions.


def get_quarter_folders(parent_dir):
    """Return sorted list of (year, quarter, folder_path) tuples that exist on disk."""
    folders = []
    for year in YEARS:
        for q in QUARTERS:
            folder_name = f"{year}q{q}_form13f"
            folder_path = os.path.join(parent_dir, folder_name)
            if os.path.isdir(folder_path):
                folders.append((year, q, folder_path))
            else:
                tqdm.write(f"  ⚠ Not found, skipping: {folder_path}")
    return folders


# DISCOVER QUARTERS

print("Scanning for quarter folders ...")
quarters = get_quarter_folders(PARENT_DIR)
print(f"  Found {len(quarters)} quarter folders\n")

if len(quarters) == 0:
    raise FileNotFoundError(
        f"No quarter folders found in '{PARENT_DIR}'. "
        "Check that PARENT_DIR is set correctly."
    )


# ACCUMULATORS

all_bank_nodes = []
all_stock_nodes = []
all_edges = []


# MAIN LOOP — one iteration per quarter

quarter_bar = tqdm(quarters, desc="Quarters", unit="quarter", ncols=80, colour="green")

for year, q, folder_path in quarter_bar:
    label = f"{year} Q{q}"
    quarter_bar.set_description(f"Processing {label}")

    sub_path = os.path.join(folder_path, "SUBMISSION.tsv")
    cov_path = os.path.join(folder_path, "COVERPAGE.tsv")
    info_path = os.path.join(folder_path, "INFOTABLE.tsv")

    # Skip quarter if any file is missing
    if not all(os.path.isfile(p) for p in [sub_path, cov_path, info_path]):
        tqdm.write(f"  ⚠ {label}: missing one or more TSV files, skipping")
        continue

    # SUBMISSION
    submission = pd.read_csv(sub_path, sep="\t", dtype=str)
    submission.columns = submission.columns.str.strip()
    submission["ACCESSION_NUMBER"] = submission["ACCESSION_NUMBER"].str.strip()

    # Tag every row with year + quarter for later reference
    submission["year"] = year
    submission["quarter"] = q

    valid_accessions = set(submission["ACCESSION_NUMBER"])
    tqdm.write(f"  {label}: {len(submission):,} filings")

    # COVERPAGE -> bank nodes
    coverpage = pd.read_csv(cov_path, sep="\t", dtype=str)
    coverpage.columns = coverpage.columns.str.strip()
    coverpage["ACCESSION_NUMBER"] = coverpage["ACCESSION_NUMBER"].str.strip()
    coverpage = coverpage[coverpage["ACCESSION_NUMBER"].isin(valid_accessions)]

    bank_df = coverpage.merge(
        submission[["ACCESSION_NUMBER", "CIK", "FILING_DATE", "year", "quarter"]],
        on="ACCESSION_NUMBER",
        how="left",
    )
    bank_df["FILING_DATE"] = pd.to_datetime(bank_df["FILING_DATE"], errors="coerce")

    bank_q = (
        bank_df.sort_values("FILING_DATE", ascending=False)
        .drop_duplicates(subset=["CIK"])[
            [
                "CIK",
                "FILINGMANAGER_NAME",
                "FILINGMANAGER_CITY",
                "FILINGMANAGER_STATEORCOUNTRY",
                "REPORTTYPE",
                "FORM13FFILENUMBER",
                "FILING_DATE",
                "ACCESSION_NUMBER",
                "year",
                "quarter",
            ]
        ]
        .rename(
            columns={
                "CIK": "bank_id",
                "FILINGMANAGER_NAME": "bank_name",
                "FILINGMANAGER_CITY": "city",
                "FILINGMANAGER_STATEORCOUNTRY": "state",
                "REPORTTYPE": "report_type",
                "FORM13FFILENUMBER": "form13f_number",
                "FILING_DATE": "filing_date",
                "ACCESSION_NUMBER": "accession_number",
            }
        )
        .reset_index(drop=True)
    )
    bank_q.insert(0, "node_type", "bank")
    all_bank_nodes.append(bank_q)

    # Build accession -> CIK lookup
    acc_to_cik = submission.set_index("ACCESSION_NUMBER")["CIK"].to_dict()
    del coverpage, bank_df, submission
    gc.collect()

    # INFOTABLE -> stock nodes + edges
    cols_to_load = [
        "ACCESSION_NUMBER",
        "NAMEOFISSUER",
        "TITLEOFCLASS",
        "CUSIP",
        "VALUE",
        "SSHPRNAMT",
        "VOTING_AUTH_SOLE",
        "VOTING_AUTH_SHARED",
        "VOTING_AUTH_NONE",
    ]

    chunks = []
    with tqdm(
        desc=f"  Reading INFOTABLE {label}",
        unit="rows",
        ncols=80,
        colour="cyan",
        leave=False,
    ) as pbar:
        for chunk in pd.read_csv(
            info_path,
            sep="\t",
            dtype=str,
            usecols=cols_to_load,
            on_bad_lines="skip",
            chunksize=CHUNKSIZE,
        ):
            chunk.columns = chunk.columns.str.strip()
            chunk["ACCESSION_NUMBER"] = chunk["ACCESSION_NUMBER"].str.strip()
            chunk = chunk[chunk["ACCESSION_NUMBER"].isin(valid_accessions)]
            if len(chunk) > 0:
                chunks.append(chunk)
            pbar.update(CHUNKSIZE)

    if not chunks:
        tqdm.write(f"  ⚠ {label}: no matching INFOTABLE rows, skipping")
        continue

    infotable = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    # Clean CUSIP
    def clean_cusip(x):
        if not isinstance(x, str):
            return None

        x = x.strip().upper()

        if x in ["", "NONE", "NULL", "NAN", "INF", "COMMON"]:
            return None

        x = x.lstrip("0")

        if len(x) < 8 or len(x) > 9:
            return None

        return x

    infotable["CUSIP"] = infotable["CUSIP"].apply(clean_cusip)
    infotable = infotable[infotable["CUSIP"].notna()]

    # Debug
    print("\n===== CUSIP CLEANING DEBUG =====")
    print("Sample cleaned CUSIPs:", infotable["CUSIP"].dropna().head(10).tolist())
    print("Remaining rows:", len(infotable))
    print("================================\n")

    # Stock nodes for this quarter
    infotable["CUSIP"] = infotable["CUSIP"].apply(clean_cusip)
    infotable = infotable[infotable["CUSIP"].notna()]

    stock_q = (
        infotable.drop_duplicates(subset=["CUSIP"])[
            ["CUSIP", "NAMEOFISSUER", "TITLEOFCLASS"]
        ]
        .rename(
            columns={
                "CUSIP": "stock_id",
                "NAMEOFISSUER": "stock_name",
                "TITLEOFCLASS": "title_of_class",
            }
        )
        .reset_index(drop=True)
    )
    stock_q.insert(0, "node_type", "stock")
    stock_q["year"] = year
    stock_q["quarter"] = q
    all_stock_nodes.append(stock_q)

    # Edges for this quarter
    infotable["bank_id"] = infotable["ACCESSION_NUMBER"].map(acc_to_cik)
    infotable = infotable.drop(
        columns=["ACCESSION_NUMBER", "NAMEOFISSUER", "TITLEOFCLASS"]
    )
    infotable = infotable.dropna(subset=["bank_id"])

    for col in [
        "VALUE",
        "SSHPRNAMT",
        "VOTING_AUTH_SOLE",
        "VOTING_AUTH_SHARED",
        "VOTING_AUTH_NONE",
    ]:
        infotable[col] = pd.to_numeric(infotable[col], errors="coerce").fillna(0)
    gc.collect()

    edges_q = (
        infotable.groupby(["bank_id", "CUSIP"], as_index=False)
        .agg(
            total_value=("VALUE", "sum"),
            total_shares=("SSHPRNAMT", "sum"),
            voting_sole=("VOTING_AUTH_SOLE", "sum"),
            voting_shared=("VOTING_AUTH_SHARED", "sum"),
            voting_none=("VOTING_AUTH_NONE", "sum"),
        )
        .rename(columns={"CUSIP": "stock_id"})
    )
    edges_q.insert(0, "edge_type", "bank_holds_stock")
    edges_q["year"] = year
    edges_q["quarter"] = q
    all_edges.append(edges_q)

    tqdm.write(f"  {label}: {len(stock_q):,} stocks, {len(edges_q):,} edges")
    del infotable, stock_q, edges_q
    gc.collect()

quarter_bar.close()


# COMBINE ALL QUARTERS

print("\nCombining all quarters ...")

with tqdm(total=3, desc="Merging", unit="table", ncols=80, colour="yellow") as mbar:

    # Bank nodes — keep most recent record per bank_id across all quarters
    bank_nodes = pd.concat(all_bank_nodes, ignore_index=True)
    bank_nodes["filing_date"] = pd.to_datetime(
        bank_nodes["filing_date"], errors="coerce"
    )
    bank_nodes = (
        bank_nodes.sort_values("filing_date", ascending=False)
        .drop_duplicates(subset=["bank_id"])
        .reset_index(drop=True)
    )
    del all_bank_nodes
    mbar.update(1)

    # Stock nodes — keep one record per CUSIP (most recently seen)
    stock_nodes = pd.concat(all_stock_nodes, ignore_index=True)
    stock_nodes = (
        stock_nodes.sort_values(["year", "quarter"], ascending=False)
        .drop_duplicates(subset=["stock_id"])
        .reset_index(drop=True)
    )
    del all_stock_nodes
    mbar.update(1)

    # Edges — keep all (one row per bank-stock-quarter combination)
    edges = pd.concat(all_edges, ignore_index=True)
    del all_edges
    gc.collect()
    mbar.update(1)


# NODE-LEVEL AGGREGATE FEATURES

print("Computing aggregate features ...")

bank_features = edges.groupby("bank_id", as_index=False).agg(
    num_stocks_held=("stock_id", "nunique"),
    total_aum_value=("total_value", "sum"),
    avg_position_size=("total_value", "mean"),
    num_quarters_active=("quarter", "nunique"),
)
bank_nodes = bank_nodes.merge(bank_features, on="bank_id", how="left")

stock_features = edges.groupby("stock_id", as_index=False).agg(
    num_holders=("bank_id", "nunique"),
    total_institutional_value=("total_value", "sum"),
    total_institutional_shares=("total_shares", "sum"),
    num_quarters_held=("quarter", "nunique"),
)
stock_nodes = stock_nodes.merge(stock_features, on="stock_id", how="left")


# SAVE AS PARQUET

print("Saving outputs ...")
files = {
    f"{OUTPUT_DIR}/nodes_bank.parquet": bank_nodes,
    f"{OUTPUT_DIR}/nodes_stock.parquet": stock_nodes,
    f"{OUTPUT_DIR}/edges_bank_stock.parquet": edges,
}
with tqdm(
    files.items(), desc="Saving", unit="file", ncols=80, colour="yellow", leave=False
) as fbar:
    for path, df in fbar:
        df.to_parquet(path, index=False)
        tqdm.write(f"  ✓ {path}  ({os.path.getsize(path)/1e6:.1f} MB)")


# SUMMARY

summary = f"""
MDGNN Graph Construction Summary
==================================
Quarters processed   : {len(quarters)} ({quarters[0][0]} Q{quarters[0][1]} — {quarters[-1][0]} Q{quarters[-1][1]})
----------------------------------
Bank nodes           : {len(bank_nodes):,}
Stock nodes          : {len(stock_nodes):,}
Bank-Stock edges     : {len(edges):,}
----------------------------------
Avg stocks per bank  : {edges.groupby('bank_id')['stock_id'].nunique().mean():.1f}
Avg banks per stock  : {edges.groupby('stock_id')['bank_id'].nunique().mean():.1f}
Most held stock      : {edges.groupby('stock_id')['bank_id'].nunique().idxmax()}
                       ({edges.groupby('stock_id')['bank_id'].nunique().max()} holders)
----------------------------------
Output files:
  {OUTPUT_DIR}/nodes_bank.parquet
  {OUTPUT_DIR}/nodes_stock.parquet
  {OUTPUT_DIR}/edges_bank_stock.parquet
"""
print(summary)
with open(f"{OUTPUT_DIR}/graph_summary.txt", "w") as f:
    f.write(summary)

print("Done!")
