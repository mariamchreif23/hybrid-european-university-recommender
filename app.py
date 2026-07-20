"""
app.py
======
Streamlit UI for the European University Recommendation System.

Deployment (Streamlit Community Cloud):
    1. Push this repo to GitHub (see README.md for the exact file list).
    2. On https://share.streamlit.io, create a new app pointing at this
       repo, branch, and app.py as the entry point.
    3. Done -- no ngrok, no Colab needed.

Run locally:
    streamlit run app.py

Data:
    Preferred: commit "university_data_with_clusters.csv" and
    "university_embeddings.npy" to the repo (produced once by
    `python build_embeddings.py`) so the app loads instantly.

    Fallback: if those two files aren't present but the raw
    "European_Universities_Final.csv" is, the app will build the
    embeddings and clusters itself on first load (slower cold start,
    cached for the rest of the session).
"""

import os
import re

import numpy as np
import pandas as pd
import streamlit as st

from recommender_logic import smart_recommend

DF_PATH = "university_data_with_clusters.csv"
EMBEDDINGS_PATH = "university_embeddings.npy"
RAW_DATA_PATH = "European_Universities_Final.csv"

st.set_page_config(
    page_title="University Recommendation System",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loading (cached so it only runs once per session, not on every rerun)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_data():
    required_columns = [
        "University", "Country", "City", "Category", "Fields_of_Study",
        "Description", "Tuition_Fees_EUR_Per_Year", "Scholarship_Availability",
        "RANK_2025", "QS_Rank_Numeric", "Overall_Score", "Website", "Cluster", "Language"
    ]

    if os.path.exists(DF_PATH) and os.path.exists(EMBEDDINGS_PATH):
        df = pd.read_csv(DF_PATH)
        X = np.load(EMBEDDINGS_PATH)
    elif os.path.exists(RAW_DATA_PATH):
        # No precomputed files -- build them now (slower first load).
        from build_embeddings import build
        df, X = build(RAW_DATA_PATH)
    else:
        st.error(
            f"Couldn't find '{DF_PATH}' + '{EMBEDDINGS_PATH}', or a raw "
            f"'{RAW_DATA_PATH}' to build them from. Add one of these to the "
            "app's repo folder."
        )
        st.stop()

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        st.error(
            f"Missing required columns in the dataset: {', '.join(missing)}. "
            "Did you run build_embeddings.py on the correct source file?"
        )
        st.stop()

    if X.shape[0] != len(df):
        st.error(
            f"Embeddings file has {X.shape[0]} rows but the dataframe has {len(df)} rows. "
            "Re-run build_embeddings.py so they match."
        )
        st.stop()

    return df, X


df, X = load_data()


@st.cache_data
def unique_fields_of_study(df: pd.DataFrame):
    """
    Fields_of_Study is a comma-separated list per row (e.g. "Engineering,
    Manufacturing, Construction, Business, Administration, Law"). This
    splits every row's list into individual field names for the sidebar
    dropdown, so the user picks one clean field rather than typing free text.
    """
    atomic = set()
    for value in df["Fields_of_Study"].dropna().unique():
        # strip parenthetical notes like "(comprehensive university...)"
        cleaned = re.sub(r"\s*\([^)]*\)\s*", "", value).strip()
        for part in cleaned.split(","):
            part = part.strip()
            if part:
                atomic.add(part)
    return sorted(atomic)


field_options = unique_fields_of_study(df)


# ---------------------------------------------------------------------------
# Sidebar -- user preferences
# ---------------------------------------------------------------------------
st.title("🎓 European University Recommendation System")
st.markdown("---")

st.sidebar.header("Your Preferences")

preferred_country = st.sidebar.selectbox(
    "Preferred Country",
    options=[""] + sorted(df["Country"].dropna().unique().tolist()),
    index=0,
    help="Leave blank for no preference.",
) or None

preferred_field_of_study = st.sidebar.selectbox(
    "Preferred Field of Study",
    options=[""] + field_options,
    index=0,
    help="Leave blank for no preference.",
) or None

tuition_choice = st.sidebar.selectbox(
    "Maximum Tuition Fee (EUR/year)",
    options=["No limit", 0, 1000, 2500, 5000, 10000, 15000, 20000],
    index=0,
    help="Choose 0 for free-tuition-only.",
)
max_tuition_fee = None if tuition_choice == "No limit" else float(tuition_choice)

scholarship_required = st.sidebar.checkbox("Require scholarship availability", value=False)

ranking_choice = st.sidebar.selectbox(
    "Maximum World Ranking (QS 2025)",
    options=["No limit", 100, 250, 500, 1000, 1400],
    index=0,
    help="Universities without a QS rank are excluded if you set a limit here.",
)
max_ranking = None if ranking_choice == "No limit" else float(ranking_choice)

recommendation_type = st.sidebar.radio(
    "Recommendation Type",
    ("Filtered (by preferences)", "Similar (hybrid: preferences + similarity)"),
)

university_name = None
if recommendation_type.startswith("Similar"):
    university_name = st.sidebar.selectbox(
        "Target university for similarity",
        options=[""] + sorted(df["University"].unique().tolist()),
        index=0,
    ) or None
    if not university_name:
        st.sidebar.warning("Pick a target university to find similar ones.")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.subheader("Your University Recommendations")

if st.sidebar.button("Get Recommendations"):
    use_similarity = recommendation_type.startswith("Similar")

    if use_similarity and not university_name:
        st.warning("Please select a target university first.")
    else:
        # --- Explain empty results instead of just showing "no matches" ---
        no_result_explanations = []
        suggestions = []
        df_for_check = df.copy()

        if preferred_country:
            df_for_check_after_country = df_for_check[df_for_check["Country"].str.contains(preferred_country, case=False, na=False, regex=False)]
            if df_for_check_after_country.empty:
                no_result_explanations.append(f"❌ No universities found in **{preferred_country}**.")
                suggestions.append("Try selecting another country or leaving the country filter empty.")
            df_for_check = df_for_check_after_country

        if not df_for_check.empty and preferred_field_of_study:
            df_for_check_after_field = df_for_check[df_for_check["Fields_of_Study"].str.contains(preferred_field_of_study, case=False, na=False, regex=False)]
            if df_for_check_after_field.empty:
                no_result_explanations.append(f"❌ No universities offer **{preferred_field_of_study}** after country filtering.")
                suggestions.append("Try selecting another field of study or leaving it empty.")
            df_for_check = df_for_check_after_field

        if not no_result_explanations and use_similarity and university_name:
            if university_name not in df["University"].values:
                no_result_explanations.append(f"❌ The selected target university '**{university_name}**' does not exist in our database.")
                suggestions.append("Try selecting another target university.")
            elif university_name not in df_for_check["University"].values:
                no_result_explanations.append(f"❌ The target university '**{university_name}**' was removed by previous filters (country/field of study). Cannot find similar universities.")
                suggestions.append("Adjust country or field of study filters, or choose a target university that matches them.")

        if not no_result_explanations and not df_for_check.empty and max_tuition_fee is not None:
            df_for_check_after_tuition = df_for_check[pd.to_numeric(df_for_check["Tuition_Fees_EUR_Per_Year"], errors="coerce") <= max_tuition_fee]
            if df_for_check_after_tuition.empty:
                no_result_explanations.append(f"❌ No universities found with tuition fees below or equal to **{max_tuition_fee} EUR** after previous filters.")
                suggestions.append("Try increasing the maximum tuition fee.")
            df_for_check = df_for_check_after_tuition

        if not no_result_explanations and not df_for_check.empty and scholarship_required:
            df_for_check_after_scholarship = df_for_check[df_for_check["Scholarship_Availability"].str.strip() == "Yes"]
            if df_for_check_after_scholarship.empty:
                no_result_explanations.append("❌ No universities offer scholarships after previous filters.")
                suggestions.append("Try removing the scholarship requirement.")
            df_for_check = df_for_check_after_scholarship

        if not no_result_explanations and not df_for_check.empty and max_ranking is not None:
            df_for_check_after_ranking = df_for_check[df_for_check["QS_Rank_Numeric"] <= max_ranking]
            if df_for_check_after_ranking.empty:
                no_result_explanations.append(f"❌ No universities ranked within **Top {int(max_ranking)}** (QS 2025) after previous filters.")
                suggestions.append("Try increasing the maximum ranking limit.")
            df_for_check = df_for_check_after_ranking

        if no_result_explanations:
            st.error("Your search is too restrictive because:")
            for exp in no_result_explanations:
                st.markdown(f"- {exp}")
            if suggestions:
                st.info("Suggestions:")
                for sug in suggestions:
                    st.markdown(f"- {sug}")
            st.stop()

        with st.spinner("Generating recommendations..."):
            results = smart_recommend(
                df=df,
                X=X,
                university_name=university_name,
                preferred_country=preferred_country,
                preferred_field_of_study=preferred_field_of_study,
                max_tuition_fee=max_tuition_fee,
                scholarship_required=scholarship_required,
                max_ranking=max_ranking,
                use_similarity=use_similarity,
                use_cluster=use_similarity,
                top_n=5,
            )

        if results.empty:
            if use_similarity and university_name:
                st.warning(f"No other universities found in the same academic cluster as '{university_name}' that match your criteria.")
                st.info("Suggestion: Try adjusting your filters, or consider turning off 'Similar' recommendation type.")
            else:
                st.warning("No universities matched your criteria. Please try broadening your search criteria.")
        else:
            st.success(f"Found {len(results)} universities matching your criteria!")
            for _, row in results.iterrows():
                st.markdown("---")
                st.header(f"🎓 {row['University']}")

                st.subheader("📍 Location")
                st.markdown(f"**Country:** {row['Country'] if pd.notna(row['Country']) else 'Not Available'}")
                st.markdown(f"**City:** {row['City'] if pd.notna(row['City']) else 'Not Available'}")

                st.subheader("🏛 Institution Information")
                st.markdown(f"**Category:** {row['Category'] if pd.notna(row['Category']) else 'Not Available'}")
                qs_rank_display = row['RANK_2025'] if pd.notna(row['RANK_2025']) and row['QS_Rank_Numeric'] != 9999 else 'Not Ranked'
                st.markdown(f"**QS World Rank (2025):** {qs_rank_display}")
                st.markdown(f"**Overall Score:** {row['Overall_Score'] if pd.notna(row['Overall_Score']) else 'Not Available'}")

                st.subheader("📚 Academic Information")
                st.markdown(f"**Fields of Study:** {row['Fields_of_Study'] if pd.notna(row['Fields_of_Study']) else 'Not Available'}")
                st.markdown(f"**Teaching Language:** {row['Language'] if pd.notna(row['Language']) else 'Not Available'}")

                st.subheader("💰 Financial Information")
                tuition_fees = row['Tuition_Fees_EUR_Per_Year']
                if pd.notna(tuition_fees):
                    st.markdown(f"**Tuition Fees:** €{int(tuition_fees):,}/year")
                else:
                    st.markdown("**Tuition Fees:** Not Available")
                st.markdown(f"**Scholarship Availability:** {row['Scholarship_Availability'] if pd.notna(row['Scholarship_Availability']) else 'Not Available'}")

                st.subheader("🌐 Official Website")
                website_url = row['Website']
                if pd.notna(website_url) and str(website_url).startswith('http'):
                    st.link_button("Visit Website", website_url)
                else:
                    st.markdown("**Website:** Not Available")

                st.subheader("📝 Description")
                st.markdown(row['Description'] if pd.notna(row['Description']) else 'Not Available')

                with st.expander("🤖 Recommendation Details and Explanation"):
                    st.markdown(f"**Cluster:** {row['Cluster']}")
                    st.markdown(f"**Overall Match Score:** {row['Final_Score_Combined']:.4f}")

                    match_points = []

                    if use_similarity:
                        match_points.append(f"• **{row['Score']:.0%} semantic similarity** to '{university_name}'.")
                        if row["Is_Same_Cluster"]:
                            match_points.append("• **Same academic cluster** as the selected university.")

                    if preferred_country and row["Matches_Country"]:
                        match_points.append(f"• Matches your preferred country: **{preferred_country}**.")

                    if preferred_field_of_study and row["Matches_Field_Of_Study"]:
                        match_points.append(f"• Offers your preferred field of study: **{preferred_field_of_study}**.")

                    if max_tuition_fee is not None and row["Matches_Tuition"]:
                        tuition_val = row['Tuition_Fees_EUR_Per_Year']
                        match_points.append(f"• Tuition (**€{int(tuition_val):,}**) is within your budget (max €{int(max_tuition_fee):,}).")

                    if scholarship_required and row["Matches_Scholarship"]:
                        match_points.append("• **Scholarships are available.**")

                    if max_ranking is not None and row["Matches_Ranking"]:
                        rank_val = row['RANK_2025'] if pd.notna(row['RANK_2025']) and row['QS_Rank_Numeric'] != 9999 else 'Not Ranked'
                        match_points.append(f"• Ranked within your requested range (QS rank: {rank_val}, max {int(max_ranking)}).")

                    if match_points:
                        st.markdown("**✅ Why this university was recommended:**")
                        for p in match_points:
                            st.markdown(p)
                    else:
                        st.markdown("No specific preferences were set to compare against.")

else:
    st.info("Set your preferences in the sidebar and click **Get Recommendations**.")
