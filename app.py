"""
app.py
======
Streamlit UI for the European University Recommendation System.

Visual design: full-screen animated hero, glassmorphism throughout, a
premium stats dashboard, elegant recommendation cards with country flags,
and a blue "AI network" color palette. All styling is plain CSS injected
once at startup -- no JS dependency, so it renders reliably on Streamlit
Community Cloud.

Deployment (Streamlit Community Cloud):
    1. Push this repo to GitHub (see README.md for the exact file list).
    2. On https://share.streamlit.io, create a new app pointing at this
       repo, branch, and app.py as the entry point.

Run locally:
    streamlit run app.py

Data:
    Preferred: commit "university_data_with_clusters.csv" and
    "university_embeddings.npy" to the repo (produced once by
    `python build_embeddings.py`) so the app loads instantly.

    Fallback: if those two files aren't present but the raw
    "European_Universities_Final.csv" is, the app will build the
    embeddings and clusters itself on first load.

Optional hero photo:
    Drop a real photo at assets/hero.jpg (any European campus you have
    the rights to use) and the hero section will use it automatically,
    with a dark gradient overlay for text contrast. Without it, the hero
    falls back to the built-in animated "constellation" background below
    -- no external image hosting required either way.
"""

import base64
import os
import re

import numpy as np
import pandas as pd
import streamlit as st

from recommender_logic import smart_recommend

DF_PATH = "university_data_with_clusters.csv"
EMBEDDINGS_PATH = "university_embeddings.npy"
RAW_DATA_PATH = "European_Universities_Final.csv"
HERO_IMAGE_PATH = "assets/hero.jpg"

st.set_page_config(
    page_title="European University Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Country -> flag emoji (superset of European + neighboring countries so any
# value in the dataset resolves; unmatched values fall back to a globe).
# ---------------------------------------------------------------------------
COUNTRY_FLAGS = {
    "United Kingdom": "🇬🇧", "UK": "🇬🇧", "Germany": "🇩🇪", "France": "🇫🇷",
    "Italy": "🇮🇹", "Spain": "🇪🇸", "Netherlands": "🇳🇱", "Belgium": "🇧🇪",
    "Switzerland": "🇨🇭", "Austria": "🇦🇹", "Sweden": "🇸🇪", "Norway": "🇳🇴",
    "Denmark": "🇩🇰", "Finland": "🇫🇮", "Ireland": "🇮🇪", "Portugal": "🇵🇹",
    "Poland": "🇵🇱", "Czech Republic": "🇨🇿", "Czechia": "🇨🇿", "Hungary": "🇭🇺",
    "Greece": "🇬🇷", "Romania": "🇷🇴", "Bulgaria": "🇧🇬", "Croatia": "🇭🇷",
    "Slovakia": "🇸🇰", "Slovenia": "🇸🇮", "Estonia": "🇪🇪", "Latvia": "🇱🇻",
    "Lithuania": "🇱🇹", "Luxembourg": "🇱🇺", "Malta": "🇲🇹", "Cyprus": "🇨🇾",
    "Iceland": "🇮🇸", "Serbia": "🇷🇸", "Ukraine": "🇺🇦", "Turkey": "🇹🇷",
    "Russia": "🇷🇺", "Belarus": "🇧🇾", "Montenegro": "🇲🇪",
    "North Macedonia": "🇲🇰", "Bosnia and Herzegovina": "🇧🇦", "Albania": "🇦🇱",
    "Moldova": "🇲🇩", "Georgia": "🇬🇪", "Armenia": "🇦🇲", "Azerbaijan": "🇦🇿",
    "Kosovo": "🇽🇰", "Liechtenstein": "🇱🇮", "Monaco": "🇲🇨",
    "San Marino": "🇸🇲", "Andorra": "🇦🇩", "Vatican City": "🇻🇦",
}


def flag_for(country):
    if pd.isna(country):
        return "🌍"
    return COUNTRY_FLAGS.get(str(country).strip(), "🌍")


# ---------------------------------------------------------------------------
# Styling -- one CSS block, injected once.
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

        :root{
            --void:#060B18; --deep:#0B1730; --panel:#0E1D3B;
            --glass:rgba(255,255,255,0.055); --glass-strong:rgba(255,255,255,0.09);
            --border:rgba(255,255,255,0.14); --border-soft:rgba(255,255,255,0.08);
            --accent:#3B82F6; --cyan:#22D3EE; --accent-soft:rgba(59,130,246,0.18);
            --text:#EAF2FF; --muted:#8DA2C4; --muted-dim:#5C7096;
        }

        html, body, [class*="css"]{ font-family:'Inter', sans-serif; }
        h1,h2,h3,h4, .stat-num, .hero-title { font-family:'Space Grotesk', sans-serif; }
        .mono, .badge, .score-tag { font-family:'JetBrains Mono', monospace; }

        [data-testid="stAppViewContainer"]{
            background:
                radial-gradient(1200px 600px at 15% -10%, rgba(59,130,246,0.16), transparent 60%),
                radial-gradient(1000px 500px at 100% 10%, rgba(34,211,238,0.10), transparent 55%),
                var(--void);
        }
        [data-testid="stHeader"]{ background:transparent; }
        .block-container{ padding-top:0 !important; max-width:1180px; }
        #MainMenu, footer{ visibility:hidden; }

        /* ---------- Sidebar: glass panel ---------- */
        [data-testid="stSidebar"]{
            background:linear-gradient(180deg, var(--deep), var(--void));
            border-right:1px solid var(--border-soft);
        }
        [data-testid="stSidebar"] .block-container{ padding-top:2rem; }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label p{
            color:var(--text) !important;
        }
        [data-testid="stSidebar"] label p{
            font-size:0.8rem; letter-spacing:.03em; text-transform:uppercase;
            color:var(--muted) !important; font-weight:600;
        }
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stTextInput input{
            background:var(--glass) !important; border:1px solid var(--border) !important;
            border-radius:10px !important; color:var(--text) !important;
        }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"]{ margin-bottom:2px; }
        [data-testid="stSidebar"] .stButton button{
            width:100%; background:linear-gradient(135deg, var(--accent), var(--cyan));
            color:#04101F; font-weight:700; border:none; border-radius:12px;
            padding:0.7rem 1rem; letter-spacing:.02em; transition:transform .15s ease, box-shadow .15s ease;
            box-shadow:0 8px 24px -8px rgba(59,130,246,0.6);
        }
        [data-testid="stSidebar"] .stButton button:hover{
            transform:translateY(-2px); box-shadow:0 12px 28px -6px rgba(34,211,238,0.55);
        }
        [data-testid="stSidebar"] hr{ border-color:var(--border-soft); }

        /* ---------- Hero ---------- */
        .hero{
            position:relative; min-height:78vh; margin:0 -1rem 0 -1rem;
            display:flex; flex-direction:column; align-items:center; justify-content:center;
            text-align:center; overflow:hidden; border-radius:0 0 32px 32px;
            background:
                linear-gradient(180deg, rgba(6,11,24,0.35) 0%, rgba(6,11,24,0.92) 92%),
                var(--hero-photo, none),
                radial-gradient(900px 500px at 20% 20%, rgba(59,130,246,0.25), transparent 60%),
                radial-gradient(700px 500px at 85% 75%, rgba(34,211,238,0.18), transparent 55%),
                linear-gradient(160deg, #081226 0%, #0A1B36 45%, #071022 100%);
            background-size:cover; background-position:center;
        }
        .constellation{ position:absolute; inset:0; opacity:0.55; pointer-events:none; }
        .node{
            position:absolute; width:3px; height:3px; border-radius:50%;
            background:var(--cyan); box-shadow:0 0 8px 2px rgba(34,211,238,0.7);
            animation:drift 14s ease-in-out infinite;
        }
        .node.b{ background:var(--accent); box-shadow:0 0 8px 2px rgba(59,130,246,0.7); }
        @keyframes drift{
            0%,100%{ transform:translate(0,0); opacity:.5; }
            50%{ transform:translate(14px,-18px); opacity:1; }
        }
        .grid-lines{
            position:absolute; inset:0;
            background-image:
                linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
            background-size:48px 48px; mask-image:radial-gradient(ellipse at center, black 40%, transparent 78%);
        }
        .eyebrow{
            font-family:'JetBrains Mono', monospace; font-size:0.78rem; letter-spacing:.18em;
            text-transform:uppercase; color:var(--cyan); z-index:2; margin-bottom:18px;
            display:flex; align-items:center; gap:10px; opacity:0; animation:fadeUp .8s ease forwards;
        }
        .eyebrow::before, .eyebrow::after{ content:""; width:28px; height:1px; background:var(--cyan); opacity:.6; }
        .hero-title{
            font-size:clamp(2.4rem, 5.4vw, 4.4rem); font-weight:700; line-height:1.06;
            color:var(--text); z-index:2; margin:0 0 20px 0; max-width:920px;
            opacity:0; animation:fadeUp .9s ease .1s forwards;
        }
        .hero-title span{
            background:linear-gradient(120deg, var(--accent), var(--cyan));
            -webkit-background-clip:text; background-clip:text; color:transparent;
        }
        .hero-sub{
            font-size:clamp(1rem, 1.5vw, 1.2rem); color:var(--muted); max-width:620px;
            z-index:2; margin-bottom:44px; line-height:1.6;
            opacity:0; animation:fadeUp .9s ease .2s forwards;
        }
        @keyframes fadeUp{ from{ opacity:0; transform:translateY(16px); } to{ opacity:1; transform:translateY(0); } }

        /* ---------- Stat strip (overlaps hero bottom edge) ---------- */
        .stat-strip{
            position:relative; z-index:3; display:grid; grid-template-columns:repeat(4,1fr);
            gap:16px; width:min(1040px, 92%); margin:-52px auto 46px auto;
            opacity:0; animation:fadeUp .9s ease .32s forwards;
        }
        .stat-card{
            background:var(--glass-strong); backdrop-filter:blur(18px); -webkit-backdrop-filter:blur(18px);
            border:1px solid var(--border); border-radius:18px; padding:20px 18px;
            text-align:center; transition:transform .2s ease, border-color .2s ease;
        }
        .stat-card:hover{ transform:translateY(-4px); border-color:rgba(34,211,238,0.5); }
        .stat-num{ font-size:1.9rem; font-weight:700; color:var(--text); line-height:1; }
        .stat-label{ font-size:0.72rem; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); margin-top:8px; }

        /* ---------- Section headers ---------- */
        .section-eyebrow{
            font-family:'JetBrains Mono', monospace; font-size:0.72rem; letter-spacing:.15em;
            text-transform:uppercase; color:var(--cyan); margin-bottom:6px;
        }
        .section-title{ font-size:1.5rem; font-weight:700; color:var(--text); margin:0 0 22px 0; }

        /* ---------- Recommendation cards ---------- */
        .uni-card{
            background:var(--glass); backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);
            border:1px solid var(--border-soft); border-radius:20px; padding:26px 28px;
            margin-bottom:20px; position:relative; overflow:hidden;
            transition:transform .2s ease, border-color .2s ease, box-shadow .2s ease;
            animation:fadeUp .6s ease forwards;
        }
        .uni-card::before{
            content:""; position:absolute; left:0; top:0; bottom:0; width:3px;
            background:linear-gradient(180deg, var(--accent), var(--cyan));
        }
        .uni-card:hover{
            transform:translateY(-3px); border-color:rgba(34,211,238,0.4);
            box-shadow:0 20px 44px -20px rgba(34,211,238,0.28);
        }
        .uni-rank{
            font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:var(--cyan);
            background:var(--accent-soft); border:1px solid rgba(34,211,238,0.3);
            border-radius:999px; padding:3px 11px; display:inline-block; margin-bottom:12px;
        }
        .uni-name{ font-size:1.35rem; font-weight:700; color:var(--text); margin:0 0 4px 0; font-family:'Space Grotesk', sans-serif; }
        .uni-loc{ color:var(--muted); font-size:0.92rem; margin-bottom:16px; }
        .badge-row{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px; }
        .badge{
            font-size:0.78rem; padding:5px 12px; border-radius:999px;
            background:var(--glass-strong); border:1px solid var(--border); color:var(--text);
            display:inline-flex; align-items:center; gap:6px;
        }
        .badge.good{ border-color:rgba(34,211,238,0.45); color:var(--cyan); }
        .uni-desc{ color:var(--muted); font-size:0.92rem; line-height:1.6; margin-bottom:6px; }

        /* ---------- Streamlit native widget re-skin (expander, alerts, link button) ---------- */
        [data-testid="stExpander"]{
            background:var(--glass); border:1px solid var(--border-soft); border-radius:14px;
            overflow:hidden; margin-top:6px;
        }
        [data-testid="stExpander"] summary{ color:var(--text) !important; font-weight:600; }
        .stAlert{ background:var(--glass-strong) !important; border:1px solid var(--border) !important; border-radius:14px !important; }
        [data-testid="stLinkButton"] a{
            background:linear-gradient(135deg, var(--accent), var(--cyan)) !important;
            color:#04101F !important; font-weight:700 !important; border:none !important; border-radius:10px !important;
        }
        hr{ border-color:var(--border-soft) !important; }

        /* ---------- Responsive ---------- */
        @media (max-width:768px){
            .stat-strip{ grid-template-columns:repeat(2,1fr); margin-top:-32px; }
            .hero{ min-height:66vh; border-radius:0 0 22px 22px; }
            .uni-card{ padding:20px 18px; }
        }

        @media (prefers-reduced-motion: reduce){
            .node, .hero-title, .hero-sub, .eyebrow, .stat-strip, .uni-card{ animation:none !important; opacity:1 !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_photo_css_var():
    """If assets/hero.jpg exists, return a CSS custom-property override that
    layers it under the gradients; otherwise the hero uses the built-in
    animated gradient/constellation background only."""
    if os.path.exists(HERO_IMAGE_PATH):
        with open(HERO_IMAGE_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"<style>:root{{ --hero-photo:url(data:image/jpeg;base64,{b64}); }}</style>"
    return ""


def render_hero(total_unis, total_countries, total_fields, ranked_count):
    nodes_html = "".join(
        f'<span class="node{" b" if i % 2 else ""}" style="left:{x}%; top:{y}%; animation-delay:{d}s;"></span>'
        for i, (x, y, d) in enumerate([
            (8, 22, 0), (18, 62, 1.2), (27, 15, 2.1), (34, 78, 0.6), (44, 40, 1.8),
            (52, 20, 0.3), (61, 68, 2.4), (70, 30, 1.0), (78, 58, 1.6), (88, 24, 0.9),
            (14, 45, 2.6), (92, 70, 1.4),
        ])
    )
    st.markdown(hero_photo_css_var(), unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="hero">
            <div class="grid-lines"></div>
            <div class="constellation">{nodes_html}</div>
            <div class="eyebrow">AI-Powered · Pan-European Matching</div>
            <div class="hero-title">Find your <span>place to study</span> in Europe</div>
            <div class="hero-sub">
                Semantic search over {total_unis:,} universities across {total_countries} countries —
                matched to your field, budget, and preferences with Sentence-BERT embeddings and
                a hybrid recommendation engine.
            </div>
        </div>
        <div class="stat-strip">
            <div class="stat-card"><div class="stat-num">{total_unis:,}</div><div class="stat-label">Universities</div></div>
            <div class="stat-card"><div class="stat-num">{total_countries}</div><div class="stat-label">Countries</div></div>
            <div class="stat-card"><div class="stat-num">{total_fields}</div><div class="stat-label">Fields of Study</div></div>
            <div class="stat-card"><div class="stat-num">{ranked_count}</div><div class="stat-label">QS Ranked</div></div>
        </div>
        """,
        unsafe_allow_html=True,
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


@st.cache_data
def unique_fields_of_study(df: pd.DataFrame):
    atomic = set()
    for value in df["Fields_of_Study"].dropna().unique():
        cleaned = re.sub(r"\s*\([^)]*\)\s*", "", value).strip()
        for part in cleaned.split(","):
            part = part.strip()
            if part:
                atomic.add(part)
    return sorted(atomic)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
inject_css()

df, X = load_data()
field_options = unique_fields_of_study(df)

render_hero(
    total_unis=len(df),
    total_countries=df["Country"].nunique(),
    total_fields=len(field_options),
    ranked_count=int((df["QS_Rank_Numeric"] != 9999).sum()),
)

# ---------------------------------------------------------------------------
# Sidebar -- user preferences
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    '<div style="font-family:\'Space Grotesk\',sans-serif; font-size:1.15rem; font-weight:700; '
    'color:#EAF2FF; margin-bottom:2px;">🎓 Your Preferences</div>'
    '<div style="color:#5C7096; font-size:0.82rem; margin-bottom:18px;">Tune your search, then get matched.</div>',
    unsafe_allow_html=True,
)

preferred_country = st.sidebar.selectbox(
    "🌍 Preferred Country",
    options=[""] + sorted(df["Country"].dropna().unique().tolist()),
    index=0,
    help="Leave blank for no preference.",
) or None

preferred_field_of_study = st.sidebar.selectbox(
    "📚 Preferred Field of Study",
    options=[""] + field_options,
    index=0,
    help="Leave blank for no preference.",
) or None

tuition_choice = st.sidebar.selectbox(
    "💶 Maximum Tuition Fee (EUR/year)",
    options=["No limit", 0, 1000, 2500, 5000, 10000, 15000, 20000],
    index=0,
    help="Choose 0 for free-tuition-only.",
)
max_tuition_fee = None if tuition_choice == "No limit" else float(tuition_choice)

scholarship_required = st.sidebar.checkbox("🎗️ Require scholarship availability", value=False)

ranking_choice = st.sidebar.selectbox(
    "🏆 Maximum World Ranking (QS 2025)",
    options=["No limit", 100, 250, 500, 1000, 1400],
    index=0,
    help="Universities without a QS rank are excluded if you set a limit here.",
)
max_ranking = None if ranking_choice == "No limit" else float(ranking_choice)

st.sidebar.markdown("---")

recommendation_type = st.sidebar.radio(
    "🧭 Recommendation Type",
    ("Filtered (by preferences)", "Similar (hybrid: preferences + similarity)"),
)

university_name = None
if recommendation_type.startswith("Similar"):
    university_name = st.sidebar.selectbox(
        "🔎 Target university for similarity",
        options=[""] + sorted(df["University"].unique().tolist()),
        index=0,
    ) or None
    if not university_name:
        st.sidebar.warning("Pick a target university to find similar ones.")

st.sidebar.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
get_recs = st.sidebar.button("✨ Get Recommendations")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="section-eyebrow">Results</div>'
    '<div class="section-title">Your University Recommendations</div>',
    unsafe_allow_html=True,
)

if get_recs:
    use_similarity = recommendation_type.startswith("Similar")

    if use_similarity and not university_name:
        st.warning("Please select a target university first.")
    else:
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

            for rank, (_, row) in enumerate(results.iterrows(), start=1):
                flag = flag_for(row["Country"])
                tuition_fees = row["Tuition_Fees_EUR_Per_Year"]
                tuition_display = f"€{int(tuition_fees):,}/yr" if pd.notna(tuition_fees) else "N/A"
                qs_rank_display = row['RANK_2025'] if pd.notna(row['RANK_2025']) and row['QS_Rank_Numeric'] != 9999 else "Not Ranked"
                scholarship_display = row['Scholarship_Availability'] if pd.notna(row['Scholarship_Availability']) else "N/A"
                language_display = row['Language'] if pd.notna(row['Language']) else "N/A"

                badges = f"""
                    <span class="badge">💶 {tuition_display}</span>
                    <span class="badge{' good' if scholarship_display == 'Yes' else ''}">🎗️ {scholarship_display}</span>
                    <span class="badge">🗣️ {language_display}</span>
                    <span class="badge">🏆 QS {qs_rank_display}</span>
                """
                if use_similarity:
                    badges += f'<span class="badge good">🤝 {row["Score"]:.0%} match</span>'

                st.markdown(
                    f"""
                    <div class="uni-card">
                        <div class="uni-rank">#{rank} RECOMMENDATION</div>
                        <div class="uni-name">{flag} {row['University']}</div>
                        <div class="uni-loc">📍 {row['City']}, {row['Country']} · {row['Category']}</div>
                        <div class="badge-row">{badges}</div>
                        <div class="uni-desc">{row['Description'] if pd.notna(row['Description']) else ''}</div>
                        <div class="uni-desc" style="margin-top:8px;"><strong style="color:#EAF2FF;">Fields:</strong> {row['Fields_of_Study'] if pd.notna(row['Fields_of_Study']) else 'Not Available'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                website_url = row['Website']
                col_a, col_b = st.columns([1, 4])
                with col_a:
                    if pd.notna(website_url) and str(website_url).startswith('http'):
                        st.link_button("🌐 Visit Website", website_url)

                with st.expander("🤖 Why this university was recommended"):
                    st.markdown(f"**Cluster:** {row['Cluster']}  ·  **Overall Match Score:** `{row['Final_Score_Combined']:.4f}`")

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
                        match_points.append(f"• Tuition (**€{int(tuition_fees):,}**) is within your budget (max €{int(max_tuition_fee):,}).")
                    if scholarship_required and row["Matches_Scholarship"]:
                        match_points.append("• **Scholarships are available.**")
                    if max_ranking is not None and row["Matches_Ranking"]:
                        match_points.append(f"• Ranked within your requested range (QS rank: {qs_rank_display}, max {int(max_ranking)}).")

                    if match_points:
                        st.markdown("**✅ Match details:**")
                        for p in match_points:
                            st.markdown(p)
                    else:
                        st.markdown("No specific preferences were set to compare against.")

else:
    st.info("Set your preferences in the sidebar and click **✨ Get Recommendations**.")
