"""
MPLADS Works Sanctioned -- Data Cleaning + Graph Construction
================================================================
Builds the MP <-> IDA fund-flow graph described in the project report.

Run with: python clean_and_build_graph.py Works_Sanctioned.csv

Outputs:
  - works_sanctioned_clean.csv   (cleaned, feature-engineered work records)
  - mplads_graph.gpickle         (networkx graph: MP + IDA nodes, work edges)
  - prints summary stats matching the report (351 MPs, 479 IDAs, 15,000 works)
"""

import sys
import re
import pickle
import pandas as pd
import networkx as nx


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop the trailing "Grand Total" row (Sr. No. is non-numeric there,
    # and its bogus value leaks into the Work Status column on export).
    df["Sr. No."] = pd.to_numeric(df["Sr. No."], errors="coerce")
    df = df.dropna(subset=["Sr. No."]).copy()
    df["Sr. No."] = df["Sr. No."].astype(int)

    # Sanction Amount -> numeric, drop unparseable rows
    df["Sanction Amount ( \u20b9 )"] = pd.to_numeric(
        df["Sanction Amount ( \u20b9 )"], errors="coerce"
    )
    df = df.dropna(subset=["Sanction Amount ( \u20b9 )"]).copy()

    # Dates -> datetime
    df["Recommended date"] = pd.to_datetime(df["Recommended date"], errors="coerce", format="mixed")
    df["Sanction Date"] = pd.to_datetime(df["Sanction Date"], errors="coerce", format="mixed")

    # Sanction lag in days (key anomaly feature: unusually fast-tracked works)
    df["sanction_lag_days"] = (df["Sanction Date"] - df["Recommended date"]).dt.days

    # The "Work" column mixes a work-ID and a category description, tab-separated
    # e.g. "WS/\t MP620/2024-2025/133166-Construction of buildings for ..."
    def split_work_id(raw: str):
        if not isinstance(raw, str):
            return None
        parts = re.split(r"[\t]", raw, maxsplit=1)
        return parts[0].strip() if parts else raw.strip()

    df["work_id"] = df["Work"].apply(split_work_id)

    # Clean IDA text (district + agency, currently one free-text field)
    df["IDA"] = df["IDA"].astype(str).str.strip()
    df["Hon'ble Members of Parliament"] = df["Hon'ble Members of Parliament"].astype(str).str.strip()

    # Categorical codes for downstream model features
    df["work_category_code"] = df["Work category"].astype("category").cat.codes
    df["work_status_code"] = df["Work Status"].astype("category").cat.codes

    keep_cols = [
        "Sr. No.", "work_id", "Work category", "work_category_code", "State",
        "IDA", "Hon'ble Members of Parliament", "Elected/Nominated",
        "Work description", "Recommended date", "Sanction Date",
        "sanction_lag_days", "Sanction Amount ( \u20b9 )", "Work Status", "work_status_code",
    ]
    df = df[keep_cols].rename(columns={
        "Hon'ble Members of Parliament": "MP",
        "Sanction Amount ( \u20b9 )": "sanction_amount",
    })
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Graph construction (MP node --work edge--> IDA node)
# ---------------------------------------------------------------------------
def build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    for mp in df["MP"].unique():
        G.add_node(("MP", mp), node_type="MP")

    for ida in df["IDA"].unique():
        G.add_node(("IDA", ida), node_type="IDA")

    for _, row in df.iterrows():
        G.add_edge(
            ("MP", row["MP"]),
            ("IDA", row["IDA"]),
            work_id=row["work_id"],
            amount=row["sanction_amount"],
            sanction_lag_days=row["sanction_lag_days"],
            status=row["Work Status"],
            category=row["Work category"],
        )
    return G


# ---------------------------------------------------------------------------
# 4. Summary stats (sanity check against report numbers)
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, G: nx.MultiDiGraph) -> None:
    n_mp = df["MP"].nunique()
    n_ida = df["IDA"].nunique()
    n_works = len(df)
    print(f"Works (edges):        {n_works}")
    print(f"Unique MPs:           {n_mp}")
    print(f"Unique IDAs:          {n_ida}")
    print(f"Graph nodes:          {G.number_of_nodes()}  (should be {n_mp + n_ida})")
    print(f"Graph edges:          {G.number_of_edges()}  (should be {n_works})")
    print(f"Amount range (INR):   {df['sanction_amount'].min():,.0f} - {df['sanction_amount'].max():,.0f}")
    print(f"Median sanction lag:  {df['sanction_lag_days'].median()} days")
    print("\nWork status breakdown:")
    print(df["Work Status"].value_counts().to_string())


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "Works_Sanctioned.csv"

    raw = load_raw(in_path)
    print(f"Loaded {len(raw)} raw rows from {in_path}\n")

    clean_df = clean(raw)
    clean_df.to_csv("works_sanctioned_clean.csv", index=False)
    print(f"Cleaned rows: {len(clean_df)} -> works_sanctioned_clean.csv\n")

    G = build_graph(clean_df)
    with open("mplads_graph.gpickle", "wb") as f:
        pickle.dump(G, f)
    print("Graph saved -> mplads_graph.gpickle\n")

    summarize(clean_df, G)
