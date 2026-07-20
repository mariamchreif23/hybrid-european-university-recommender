"""
recommender_logic.py
=====================
Core recommendation function for the European University Recommendation
System. Kept independent from Streamlit / notebook-specific code so it can
be imported and unit-tested on its own.

Notes:
  - Uses the column name "Fields_of_Study" (plural) everywhere.
  - Ranking filters/sorts use "QS_Rank_Numeric" (a clean int column,
    9999 = not ranked) instead of doing pd.to_numeric() on "RANK_2025",
    which is a TEXT column containing values like "Not Ranked" and QS band
    ranges like "601-650" -- to_numeric(..., errors="coerce") turns ALL of
    those into NaN, silently breaking both the ranking filter and the
    Matches_Ranking flag. "RANK_2025" is kept only for display.
  - Scholarship matching checks for the exact "Yes" category instead
    of a substring "Yes" search that depended on incidental wording.
  - `df` and `X` (embeddings matrix) are explicit parameters, not globals
    -- avoids the "works in the notebook, breaks in the Streamlit app"
    class of bugs.
  - Country / field-of-study text matching uses regex=False, so a user
    typing something like "C++ " or "(Design)" can't blow up
    str.contains() with an invalid regex.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

RESULT_COLUMNS = [
    "University", "Country", "City", "Category", "Fields_of_Study",
    "Description", "Language", "Tuition_Fees_EUR_Per_Year", "Scholarship_Availability",
    "RANK_2025", "QS_Rank_Numeric", "Overall_Score", "Website", "Cluster",
    "Score", "Is_Same_Cluster",
    "Matches_Country", "Matches_Field_Of_Study", "Matches_Tuition",
    "Matches_Scholarship", "Matches_Ranking",
    "Final_Score_Combined",
]


def smart_recommend(
    df,
    X,
    university_name=None,
    preferred_country=None,
    preferred_field_of_study=None,
    max_tuition_fee=None,
    scholarship_required=False,
    max_ranking=None,
    top_n=5,
    use_similarity=True,
    use_cluster=True,
    weight_similarity=1.0,
    weight_cluster=0.6,
    weight_field_of_study=0.4,
    weight_tuition=0.3,
    weight_scholarship=0.1,
    weight_ranking=0.5,
):
    """
    Recommend universities either by:
      (a) pure preference filtering (use_similarity=False), or
      (b) a hybrid of preference filtering + embedding similarity +
          cluster membership, relative to a chosen `university_name`.

    Returns a DataFrame with RESULT_COLUMNS, or an empty DataFrame if no
    universities match.
    """
    required_cols = {"University", "Country", "Fields_of_Study",
                      "Tuition_Fees_EUR_Per_Year", "Scholarship_Availability",
                      "RANK_2025", "QS_Rank_Numeric"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s) in df: {sorted(missing)}")

    working_df = df.copy()

    # ---------------- Hard filters (drop rows that don't qualify) ----------------
    if preferred_country:
        working_df = working_df[
            working_df["Country"].str.contains(preferred_country, case=False, na=False, regex=False)
        ]
        if working_df.empty:
            return pd.DataFrame(columns=RESULT_COLUMNS)

    if preferred_field_of_study:
        working_df = working_df[
            working_df["Fields_of_Study"].str.contains(preferred_field_of_study, case=False, na=False, regex=False)
        ]
        if working_df.empty:
            return pd.DataFrame(columns=RESULT_COLUMNS)

    if max_tuition_fee is not None:
        working_df = working_df[
            pd.to_numeric(working_df["Tuition_Fees_EUR_Per_Year"], errors="coerce") <= max_tuition_fee
        ]
        if working_df.empty:
            return pd.DataFrame(columns=RESULT_COLUMNS)

    if scholarship_required:
        working_df = working_df[working_df["Scholarship_Availability"].str.strip() == "Yes"]
        if working_df.empty:
            return pd.DataFrame(columns=RESULT_COLUMNS)

    if max_ranking is not None:
        # QS_Rank_Numeric is always a clean int (real rank, or 9999 if unranked)
        working_df = working_df[working_df["QS_Rank_Numeric"] <= max_ranking]
        if working_df.empty:
            return pd.DataFrame(columns=RESULT_COLUMNS)

    # ---------------- Explainability flags (for the "why recommended" UI) --------
    working_df["Matches_Country"] = (
        working_df["Country"].str.contains(preferred_country, case=False, na=False, regex=False)
        if preferred_country else False
    )
    working_df["Matches_Field_Of_Study"] = (
        working_df["Fields_of_Study"].str.contains(preferred_field_of_study, case=False, na=False, regex=False)
        if preferred_field_of_study else False
    )
    working_df["Matches_Tuition"] = (
        pd.to_numeric(working_df["Tuition_Fees_EUR_Per_Year"], errors="coerce") <= max_tuition_fee
        if max_tuition_fee is not None else False
    )
    working_df["Matches_Scholarship"] = (
        (working_df["Scholarship_Availability"].str.strip() == "Yes")
        if scholarship_required else False
    )
    working_df["Matches_Ranking"] = (
        (working_df["QS_Rank_Numeric"] <= max_ranking)
        if max_ranking is not None else False
    )

    # ---------------- Mode 1: preference-only filtering (no similarity) -----------
    if not use_similarity or not university_name:
        working_df["Is_Same_Cluster"] = False
        working_df["Score"] = 0.0

        working_df["Final_Score_Combined"] = (
            weight_field_of_study * working_df["Matches_Field_Of_Study"].astype(int)
            + weight_tuition * working_df["Matches_Tuition"].astype(int)
            + weight_scholarship * working_df["Matches_Scholarship"].astype(int)
            + weight_ranking * working_df["Matches_Ranking"].astype(int)
        )

        working_df = working_df.sort_values(
            by=["Final_Score_Combined", "University"], ascending=[False, True]
        )
        return (
            working_df.drop_duplicates(subset=["University"])[RESULT_COLUMNS]
            .head(top_n)
            .reset_index(drop=True)
        )

    # ---------------- Mode 2: hybrid similarity-based recommendation -------------
    if university_name not in df["University"].values:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    target_idx = df.index[df["University"] == university_name][0]
    target_embedding = X[target_idx].reshape(1, -1)
    target_cluster = df.loc[target_idx, "Cluster"]

    if use_cluster:
        candidates = working_df[working_df["Cluster"] == target_cluster].copy()
    else:
        candidates = working_df.copy()

    candidates = candidates[candidates["University"] != university_name].copy()
    if candidates.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    candidate_indices = candidates.index.tolist()
    sim_scores = cosine_similarity(target_embedding, X[candidate_indices])[0]
    candidates["Score"] = sim_scores
    candidates["Is_Same_Cluster"] = candidates["Cluster"] == target_cluster

    candidates["Final_Score_Combined"] = (
        weight_similarity * candidates["Score"]
        + weight_cluster * candidates["Is_Same_Cluster"].astype(int)
        + weight_field_of_study * candidates["Matches_Field_Of_Study"].astype(int)
        + weight_tuition * candidates["Matches_Tuition"].astype(int)
        + weight_scholarship * candidates["Matches_Scholarship"].astype(int)
        + weight_ranking * candidates["Matches_Ranking"].astype(int)
    )

    candidates = candidates.sort_values(by="Final_Score_Combined", ascending=False)
    return candidates[RESULT_COLUMNS].head(top_n).reset_index(drop=True)
