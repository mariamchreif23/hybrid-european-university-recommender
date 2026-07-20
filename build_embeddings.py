
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

DATA_PATH = "/content/European_Universities_Final .csv"
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


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} universities from {DATA_PATH}")

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


if __name__ == "__main__":
    main()
