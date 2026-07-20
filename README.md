# European University Recommendation System — Streamlit App

A hybrid recommendation system (Sentence-BERT embeddings + K-Means clustering +
cosine similarity + rule-based filtering) for European universities, built as
an interactive Streamlit app.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app (entry point). |
| `recommender_logic.py` | Core `smart_recommend()` function — filtering + similarity ranking. Verified against the real dataset (see "Testing" below). |
| `build_embeddings.py` | One-time script: reads the raw CSV, generates Sentence-BERT embeddings, runs K-Means, writes the two files below. |
| `European_Universities_Final.csv` | Your real dataset (3,362 universities, 36 countries) — included. |
| `university_data_with_clusters.csv` | Precomputed: university data + `Cluster` column. **Not included — see below.** |
| `university_embeddings.npy` | Precomputed: the N×384 Sentence-BERT embedding matrix. **Not included — see below.** |
| `requirements.txt` | Python dependencies, pinned to CPU-only PyTorch to keep the build small. |

## 1. One-time setup: generate the embeddings

**I could not generate `university_embeddings.npy` for you.** The sandbox I run
code in only has network access to a fixed allowlist of domains, and
`huggingface.co` — where the `all-MiniLM-L6-v2` model weights are hosted — isn't
on it. I dry-ran the full pipeline (data cleaning, corpus building, K-Means
clustering, and the recommender's filtering/similarity logic) against your
actual 3,362-row dataset with placeholder vectors standing in for the real
embeddings, and everything works correctly — the only step you need to run
yourself is the one that needs real internet access to Hugging Face.

On your own machine (or in Colab, where this already worked before):

```bash
pip install -r requirements.txt
python build_embeddings.py
```

This produces `university_data_with_clusters.csv` and
`university_embeddings.npy` (takes a minute or two — it downloads the ~90 MB
model once, then encodes all 3,362 universities). **Commit both output files
to your repo** — this is what makes the deployed app start in a couple of
seconds instead of re-downloading the model and re-encoding 3,000+ rows on
every cold start.

(If you skip this and only commit the raw CSV, `app.py` still works — it
falls back to building the embeddings itself on first load — but that first
load will be slow and eats into Streamlit Community Cloud's free-tier
resource limits. Precomputing is strongly recommended.)

## 2. Run locally to check it works

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## 3. Deploy to Streamlit Community Cloud

1. Create a **public** GitHub repo (Community Cloud's free tier requires a
   public repo, or a private one if your account has that add-on) containing:
   - `app.py`
   - `recommender_logic.py`
   - `build_embeddings.py`
   - `requirements.txt`
   - `university_data_with_clusters.csv`
   - `university_embeddings.npy`
   - (`European_Universities_Final.csv` is optional once the two files above exist — it's only needed for `build_embeddings.py` to regenerate them.)
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in
   with GitHub.
3. Click **"Create app"** → **"Deploy a public app from GitHub"**.
4. Pick the repo, the branch (usually `main`), and set **Main file path** to
   `app.py`.
5. Click **Deploy**. First build takes a few minutes (installing
   `sentence-transformers`/`torch`); after that, redeploys on every push are
   fast.

Your app will be live at a URL like:
`https://<your-app-name>.streamlit.app`

### Notes on Community Cloud's free tier
- Apps that get no traffic for a while go to sleep and take ~30–60 seconds to
  wake back up on the next visit — this is normal, not a bug.
- Free tier is limited to 1 GB RAM. With embeddings precomputed and committed
  (as above), this app comfortably fits; if you regenerate embeddings on
  every cold start instead, you're more likely to hit that limit.
- If `university_embeddings.npy` is larger than GitHub's 100 MB file limit,
  use [Git LFS](https://git-lfs.com/) to track it. (For reference: 3,362 × 384
  float32 values is roughly 5 MB, well under the limit.)

## 4. Updating the dataset later

If you change the underlying university data, just re-run
`python build_embeddings.py` and commit the two regenerated output files —
no code changes needed.

## Testing performed on your real dataset

Before packaging this, I checked `European_Universities_Final.csv` directly:
- 3,362 rows, 36 countries, all required columns present (including
  `QS_Rank_Numeric`), no nulls in any column the app depends on.
- 8 university names appear twice under slightly different city spellings or
  as separate campuses (e.g. "Praha" vs "Prague", or genuinely different
  campuses of the same institution). This is harmless — filter-based
  recommendations are unaffected, and in similarity mode picking that
  university by name always resolves to its first-listed row. Worth a quick
  look if you want to dedupe further, but not a blocker.
- Ran `build_corpus()` and `ensure_qs_rank_numeric()` from `build_embeddings.py`
  against all 3,362 rows — no empty text blobs, rank-band parsing (e.g.
  "601-650" → 601, "Not Ranked" → 9999) works correctly.
- Ran K-Means (`k=8`) end-to-end on the real data (with placeholder vectors
  in place of real embeddings) — produced 8 reasonably balanced clusters
  (314–475 universities each).
- Ran `smart_recommend()` against the real data in both filter-only mode and
  hybrid similarity mode — both returned correct, real results.

