"""
build_embeddings.py
====================
One-time (or "run when the data changes") preprocessing script.

Reads the raw university CSV, builds a Sentence-BERT embedding for each
university, clusters the embeddings with K-Means, and writes:
  - university_embeddings.npy         (N x 384 float32 matrix)
  - university_data_with_clusters.csv (original data + Cluster column)

app.py loads those two output files directly, so it never has to touch
the Sentence-Transformers model at request time -- this keeps Streamlit
Community Cloud cold starts fast and avoids re-downloading/re-encoding
on every reboot.

Usage:
    python build_embeddings.py

Run this locally once (or in Colab), then commit the two output files
to the repo alongside app.py before deploying to Streamlit Community Cloud.
"""

import numpy as np
import pandas as pd

DATA_PATH = "European_Universities_Final.csv"
EMBEDDINGS_OUT = "university_embeddings.npy"
DF_OUT = "university_data_with_clusters.csv"
N_CLUSTERS = 8  # chosen from prior silhouette / Davies-Bouldin / Calinski-Harabasz analysis


def build_corpus(df: pd.DataFrame) -> pd.Series:
    """
    One text blob per university, combining the fields that actually carry
    subject-matter / character information. Country and City are included
    so the embedding also reflects geography, which the "similar
    universities" recommendation implicitly leans on.
    """
    return (
        "University: " + df["University"].fillna("") + ". "
        + "Country: " + df["Country"].fillna("") + ". "
        + "City: " + df["City"].fillna("") + ". "
        + "Category: " + df["Category"].fillna("") + ". "
        + "Fields of study: " + df["Fields_of_Study"].fillna("") + ". "
        + "Description: " + df["Description"].fillna("")
    )


def ensure_qs_rank_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    recommender_logic.py needs a clean numeric ranking column. The source
    CSV is expected to already contain "QS_Rank_Numeric" (9999 = not
    ranked). If it's missing for any reason, derive it from "RANK_2025",
    which is text and may contain values like "Not Ranked" or band ranges
    such as "601-650" -- in that case we take the lower bound of the band.
    """
    if "QS_Rank_Numeric" in df.columns:
        return df

    if "RANK_2025" not in df.columns:
        raise ValueError(
            "Dataset has neither 'QS_Rank_Numeric' nor 'RANK_2025' -- "
            "cannot build a numeric ranking column."
        )

    def parse_rank(value):
        if pd.isna(value):
            return 9999
        text = str(value).strip()
        if not text or text.lower() in {"not ranked", "unranked", "n/a", "na"}:
            return 9999
        first_number = text.split("-")[0].replace("+", "").strip()
        try:
            return int(first_number)
        except ValueError:
            return 9999

    df = df.copy()
    df["QS_Rank_Numeric"] = df["RANK_2025"].apply(parse_rank)
    return df


def build(data_path: str = DATA_PATH):
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} universities from {data_path}")

    df = ensure_qs_rank_numeric(df)
    corpus = build_corpus(df)

    print("Loading Sentence-Transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding university text into embeddings...")
    X = model.encode(corpus.tolist(), show_progress_bar=True, batch_size=64)
    X = np.asarray(X)

    print(f"Clustering into {N_CLUSTERS} clusters with K-Means...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X)

    np.save(EMBEDDINGS_OUT, X)
    df.to_csv(DF_OUT, index=False)

    print(f"Saved embeddings -> {EMBEDDINGS_OUT}  (shape={X.shape})")
    print(f"Saved dataframe with clusters -> {DF_OUT}  (shape={df.shape})")
    return df, X


if __name__ == "__main__":
    build()
