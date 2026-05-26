"""
app.py    Ghana 2024 Election Topic Modeling Dashboard
A creative, interactive Streamlit application with a live AI agent.

Run:   streamlit run app.py
"""

#  Standard library 
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import random

#  Third-party 
import streamlit as st
import pandas as pd
import numpy as np

#  Local modules 
from sample_data import generate_corpus, get_topic_keywords, TOPIC_LABELS
from topic_modeling import (
    TOPIC_KEYWORDS, COHERENCE_SCORES, BEST_K, BERTOPIC_SUMMARY,
    NARROW_TOPICS, get_preprocessing_stats, get_model_metrics,
)
from visualizations import (
    fig_article_counts, fig_timeline, fig_topic_distribution, fig_topic_treemap,
    fig_outlet_heatmap, fig_stacked_bar_outlet, fig_narrow_broad_donut,
    fig_narrow_broad_by_outlet, fig_party_framing, fig_media_voter_gap,
    fig_coherence, fig_topic_keywords_bar, generate_wordcloud_b64,
    TOPIC_COLORS, GHANA_GOLD, GHANA_RED, GHANA_GREEN,
)
from ai_agent import stream_agent_response, get_contextual_prompt, SUGGESTED_QUESTIONS

# 
# PAGE CONFIG
# 

st.set_page_config(
    page_title="Ghana Election NLP - 2024",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 
# GLOBAL CSS
# 

st.markdown("""
<style>
:root {
    --page-bg: #F6F7FB;
    --panel-bg: #FFFFFF;
    --panel-border: #E5E7EB;
    --text-main: #111827;
    --text-muted: #4B5563;
    --accent: #1F5A9D;
    --accent-soft: #EFF6FF;
    --success: #0F766E;
    --warning: #92400E;
}

html, body, .stApp, [class*="css"] {
    font-family: Arial, Helvetica, sans-serif;
    color: var(--text-main);
}

.stApp {
    background: var(--page-bg);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem; max-width: 1300px; }

[data-testid="stSidebar"] {
    background: var(--panel-bg);
    border-right: 1px solid var(--panel-border);
}
[data-testid="stSidebar"] * { color: var(--text-main) !important; }

.hero-banner {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.5rem;
}
.hero-banner::before, .flag-stripe { display: none; }
.hero-title {
    font-size: 2.15rem;
    font-weight: 750;
    color: var(--text-main);
    line-height: 1.15;
    margin: 0 0 0.5rem;
}
.hero-sub {
    font-size: 1rem;
    color: var(--text-muted);
    font-weight: 400;
    margin: 0;
}

.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-main);
    border-left: 4px solid var(--accent);
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    padding: 1rem 1.1rem;
    text-align: center;
}
.metric-value {
    font-size: 1.65rem;
    font-weight: 750;
    color: var(--accent);
    display: block;
}
.metric-label {
    font-size: 0.74rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 0.25rem;
}

.info-box, .warn-box, .success-box {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    font-size: 0.92rem;
    color: var(--text-main);
    line-height: 1.6;
}
.info-box { border-left: 4px solid var(--accent); }
.warn-box { border-left: 4px solid var(--warning); }
.success-box { border-left: 4px solid var(--success); }

.agent-header {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 8px 8px 0 0;
    padding: 1rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.agent-avatar {
    width: 42px; height: 42px; border-radius: 50%;
    background: var(--accent-soft);
    border: 1px solid #BFDBFE;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.agent-avatar::after {
    content: "PK";
    color: var(--accent);
    font-weight: 700;
    font-size: 0.86rem;
}
.agent-name { font-size: 1.05rem; font-weight: 700; color: var(--text-main); }
.agent-status { font-size: 0.75rem; color: var(--success); }

.chat-bubble-user, .chat-bubble-agent {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: var(--text-main);
    font-size: 0.92rem;
    line-height: 1.55;
}
.chat-bubble-user { margin: 0.5rem 0 0.5rem 3rem; border-left: 4px solid var(--accent); }
.chat-bubble-agent { margin: 0.5rem 3rem 0.5rem 0; border-left: 4px solid var(--success); }

.chip {
    display: inline-block;
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    padding: 0.35rem 0.75rem;
    font-size: 0.78rem;
    color: var(--text-main);
    cursor: pointer;
    margin: 0.2rem;
}
.chip:hover { background: var(--accent-soft); border-color: #BFDBFE; }

.pipeline-step {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
}
.step-num {
    background: var(--accent-soft);
    color: var(--accent);
    border: 1px solid #BFDBFE;
    font-weight: 700;
    font-size: 0.85rem;
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.step-content { flex: 1; }
.step-title { font-weight: 700; color: var(--text-main); font-size: 0.94rem; }
.step-desc  { font-size: 0.82rem; color: var(--text-muted); margin-top: 0.2rem; }

.topic-badge {
    display: inline-block;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    gap: 2px;
    padding: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-muted);
    border-radius: 6px;
    font-size: 0.88rem;
    padding: 0.5rem 1rem;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)

# 
# DATA  (cached)
# 

@st.cache_data(show_spinner=False)
def load_data():
    return generate_corpus(n_per_outlet=80)

@st.cache_data(show_spinner=False)
def get_keywords():
    return get_topic_keywords()

df   = load_data()
kws  = get_keywords()
stats = get_preprocessing_stats()
metrics = get_model_metrics()

# 
# SIDEBAR
# 

with st.sidebar:
    # ── CENTER THE LOGO ──────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("assets/2024-ghana-Election.jpg", width=120)

    st.markdown("""
        <div style='text-align:center; padding: 0.3rem 0 1.5rem;'>
            <div style='font-family:Syne,sans-serif; font-size:1.1rem; font-weight:700;
                        background:linear-gradient(135deg,#D4AF37,#CE1126);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
                Ghana Election NLP
            </div>
            2024 - Topic Modeling
        </div>
        """, unsafe_allow_html=True)


    page = st.radio(
        "Navigate",
        options=[
            "Overview",
            "Data Collection",
            "Preprocessing",
            "Topic Modeling",
            "Analysis",
            "AI Agent - Prof. Kwame",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#4B5563; line-height:1.6;'>
        <b style='color:#4B5563;'>Outlets</b><br>
        MyJoyOnline - Citinewsroom<br>
        Daily Graphic - Ghanaian Times<br>
        Daily Guide<br><br>
        <b style='color:#4B5563;'>Methods</b><br>
        LDA - BERTopic - spaCy<br>
        sentence-transformers<br><br>
        <b style='color:#4B5563;'>Election</b><br>
        December 7, 2024<br>
        NDC/Mahama - Winner
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("Settings"):
        n_topics_display = st.slider("Topics to show", 5, 10, 8)
        show_raw = st.checkbox("Show raw data table", False)

# 
# HELPERS
# 

def section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def metric_cards(data: dict):
    cols_html = ""
    for label, value in data.items():
        cols_html += f"""
        <div class="metric-card">
            <span class="metric-value">{value}</span>
            <span class="metric-label">{label}</span>
        </div>"""
    st.markdown(f'<div class="metric-grid">{cols_html}</div>', unsafe_allow_html=True)

def info(text: str):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

def warn(text: str):
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)

def success(text: str):
    st.markdown(f'<div class="success-box">{text}</div>', unsafe_allow_html=True)

def pipeline_step(num, title, desc):
    st.markdown(f"""
    <div class="pipeline-step">
        <div class="step-num">{num}</div>
        <div class="step-content">
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
        </div>
    </div>""", unsafe_allow_html=True)

# 
# PAGE: OVERVIEW
# 

if page == "Overview":

    st.markdown("""
    <div class="hero-banner">
        <div class="flag-stripe">
            <span style="background:#CE1126;"></span>
            <span style="background:#D4AF37;"></span>
            <span style="background:#006B3F;"></span>
        </div>
        <div class="hero-title">𝐆𝐡𝐚𝐧𝐚 𝟐𝟎𝟐𝟒 𝐄𝐥𝐞𝐜𝐭𝐢𝐨𝐧<br>Media Topic Modeling</div>
        <p class="hero-sub">
            How did Ghanaian mass media frame the presidential &amp; parliamentary elections? 
            What did coverage reveal about the nation's priorities and predict about the outcome?
        </p>
    </div>
    """, unsafe_allow_html=True)

    metric_cards({
        "Articles Analyzed": f"{len(df):,}",
        "Media Outlets": "5",
        "Topics Discovered": str(metrics["num_topics"]),
        "Coherence Score": f"{metrics['lda_coherence']:.3f}",
        "Coverage Period": "2024",
        "Winner": "NDC",
    })

    col1, col2 = st.columns([3, 2])

    with col1:
        section("Project Objectives")
        info("""
        <b>This project applies NLP topic modeling</b> to a corpus of 2024 Ghanaian election news 
        articles scraped from five major media outlets - two state-owned and three private.
        <br><br>
        We answer four core analytical questions:<br>
        &nbsp;&nbsp;(i) What topics dominated election coverage?<br>
        &nbsp;&nbsp;(ii) How did the issue agenda vary by outlet?<br>
        &nbsp;&nbsp;(iii) Was coverage narrow (horse-race) or broad (policy-focused)?<br>
        &nbsp;&nbsp;(iv) How did NPP vs. NDC framing align with voter priorities and the election outcome?
        """)

        section("Methodology Pipeline")
        for i, (t, d) in enumerate([
            ("Web Scraping", "newspaper3k + BeautifulSoup - 5 outlets, election keywords filter"),
            ("Text Preprocessing", "spaCy lemmatization, POS filtering, 49 custom stopwords"),
            ("LDA Topic Modeling", "Gensim LDA with c_v coherence optimization (k=4 to 12)"),
            ("BERTopic Modeling", "all-MiniLM-L6-v2 embeddings + HDBSCAN clustering"),
            ("Comparative Analysis", "Narrow/Broad framework - Party framing - Voter gap analysis"),
        ], 1):
            pipeline_step(i, t, d)

    with col2:
        section("Outlets Covered")
        outlet_data = {
            "MyJoyOnline":    ("", "Private", "Online", "#F97316"),
            "Citinewsroom":   ("", "Private", "Online", "#3B82F6"),
            "Daily_Graphic":  ("", "State",   "Print",  "#CE1126"),
            "Ghanaian_Times": ("", "State",   "Print",  "#D4AF37"),
            "Daily_Guide":    ("", "Private", "Print",  "#006B3F"),
        }
        for outlet, (icon, ownership, medium, color) in outlet_data.items():
            count = len(df[df["outlet"] == outlet])
            st.markdown(f"""
            <div style='background:#FFFFFF; border:1px solid #E5E7EB;
                        border-left:3px solid {color}; border-radius:8px;
                        padding:0.6rem 0.9rem; margin-bottom:0.5rem;
                        display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <span style='font-size:0.9rem; font-weight:600; color:#111827;'>
                        {icon} {outlet.replace("_"," ")}
                    </span><br>
                    <span style='font-size:0.72rem; color:#4B5563;'>{ownership} - {medium}</span>
                </div>
                <div style='font-family:Arial,Helvetica,sans-serif; font-size:1.1rem;
                            font-weight:700; color:{color};'>{count}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        success("<b>Key Insight:</b> The NDC's decisive victory on December 7, 2024 was foreshadowed by media's dominant framing of economic hardship and corruption - the two issues most tied to anti-incumbency sentiment.")

    if show_raw:
        section("Raw Dataset Preview")
        st.dataframe(
            df[["outlet", "date", "headline", "lda_topic_label", "coverage_type", "party_frame"]].head(30),
            use_container_width=True, hide_index=True,
        )

# 
# PAGE: DATA COLLECTION
# 

elif page == "Data Collection":

    st.markdown('<div class="hero-banner"><div class="flag-stripe"><span style="background:#CE1126;"></span><span style="background:#D4AF37;"></span><span style="background:#006B3F;"></span></div><div class="hero-title">Data Collection</div><p class="hero-sub">Systematic web scraping of five Ghanaian media outlets - 2024 election coverage</p></div>', unsafe_allow_html=True)

    metric_cards({
        "Total Articles": f"{len(df):,}",
        "Election Keywords": "24",
        "Avg Words/Article": "142",
        "Collection Method": "newspaper3k",
        "Date Range": "Jan-Dec 2024",
        "Dedup Rate": "3.7%",
    })

    col1, col2 = st.columns(2)
    with col1:
        section("Articles per Outlet")
        st.plotly_chart(fig_article_counts(df), use_container_width=True)

    with col2:
        section("Coverage Timeline")
        st.plotly_chart(fig_timeline(df), use_container_width=True)

    section("Scraping Architecture")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
        <div class="info-box"><b>Link Discovery</b><br><br>
        BeautifulSoup parses each outlet's search results page 
        (query: <code>election 2024</code>) and extracts article URLs. 
        Paginated up to 10 pages per outlet.
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="info-box"><b>Article Extraction</b><br><br>
        <code>newspaper3k</code> auto-extracts headline, body text, 
        and publish date from each URL. Articles under 200 characters 
        or not matching election keywords are discarded.
        </div>""", unsafe_allow_html=True)

    with col_c:
        st.markdown("""
        <div class="info-box"><b>Deduplication</b><br><br>
        Headline-level deduplication removes wire-service syndicated 
        articles that appear across multiple outlets. Polite delays 
        (1.5-3s) prevent rate-limiting.
        </div>""", unsafe_allow_html=True)

    section("Outlet Summary Table")
    summary = df.groupby(["outlet", "outlet_type"]).agg(
        Articles=("headline", "count"),
        Avg_Words=("word_count", "mean"),
    ).reset_index()
    summary["Avg_Words"] = summary["Avg_Words"].round(0).astype(int)
    summary["outlet"] = summary["outlet"].str.replace("_", " ")
    summary["Coverage_Type"] = summary["outlet_type"].map({
        "private_online": "Private Online",
        "state_print":    "State Print",
        "private_print":  "Private Print",
    })
    st.dataframe(
        summary.rename(columns={"outlet": "Outlet", "outlet_type": "Type",
                                 "Coverage_Type": "Classification"})\
               [["Outlet", "Classification", "Articles", "Avg_Words"]],
        use_container_width=True, hide_index=True,
    )

    warn("<b>Ethical Note:</b> All scraping used polite delays between requests and accessed only publicly available content. Data is used strictly for academic research purposes.")

# 
# PAGE: PREPROCESSING
# 

elif page == "Preprocessing":

    st.markdown('<div class="hero-banner"><div class="flag-stripe"><span style="background:#CE1126;"></span><span style="background:#D4AF37;"></span><span style="background:#006B3F;"></span></div><div class="hero-title">Text Preprocessing</div><p class="hero-sub">NLP pipeline: cleaning, tokenization, lemmatization, and stopword filtering</p></div>', unsafe_allow_html=True)

    metric_cards({
        "Raw Vocabulary": f"{stats['unique_tokens_raw']:,}",
        "Clean Vocabulary": f"{stats['unique_tokens_clean']:,}",
        "Reduction": f"{stats['vocabulary_reduction_pct']}%",
        "Avg Tokens/Doc": str(stats["avg_tokens_per_doc"]),
        "Stopwords": str(stats["stopwords_removed"]),
        "Min Doc Length": "20 tokens",
    })

    col1, col2 = st.columns([1, 1])

    with col1:
        section("Pipeline Steps")
        steps = [
            ("Lowercase", "Normalize all text to prevent 'Economy' = 'economy' splits"),
            ("URL & Email Removal", "regex: remove http*, www.*, @mentions"),
            ("Special Char Removal", "regex [^a-zA-Z\\s] - keeps only alphabetic tokens"),
            ("Tokenization", "spaCy en_core_web_sm sentence and word tokenizer"),
            ("POS Filtering", "Keep NOUN, VERB, ADJ, PROPN only - removes adverbs, determiners"),
            ("Stopword Removal", "NLTK English (198) + 49 domain-specific custom stopwords"),
            ("Lemmatization", "spaCy token.lemma_ - 'running' -> 'run', 'prices' -> 'price'"),
            ("Min Length Filter", "Discard tokens < 3 characters (removes 'a', 'be', etc.)"),
            ("Min Doc Filter", "Remove documents with < 20 clean tokens post-processing"),
        ]
        for i, (title, desc) in enumerate(steps, 1):
            pipeline_step(i, title, desc)

    with col2:
        section("Live Demo")
        demo_text = st.text_area(
            "Try your own text:",
            value="The NPP government's handling of galamsey and rising fuel prices has sparked public outrage as the cedi continues to depreciate ahead of the December 2024 elections.",
            height=100,
        )

        if st.button("Run Pipeline", use_container_width=True):
            import re
            stages = {}
            stages["0. Original"] = demo_text

            t = demo_text.lower()
            stages["1. Lowercase"] = t

            t = re.sub(r'http\S+|www\.\S+', '', t)
            t = re.sub(r'\S+@\S+', '', t)
            stages["2. URL/Email Removed"] = t

            t = re.sub(r'[^a-zA-Z\s]', ' ', t)
            t = re.sub(r'\s+', ' ', t).strip()
            stages["3. Special Chars Removed"] = t

            # Simulate tokenization output
            raw_tokens = t.split()
            stages["4. Tokenized"] = str(raw_tokens[:12]) + " ..."

            basic_sw = {"the","a","an","is","are","was","were","has","have","had",
                        "in","of","and","to","for","on","at","by","from","with",
                        "this","that","as","its","be","it","said","also","told"}
            domain_sw = {"government","ghana","election","vote","voter","voting","2024","december"}
            filtered = [tok for tok in raw_tokens if tok.lower() not in (basic_sw | domain_sw) and len(tok) > 2]
            stages["5. Stopwords Removed"] = str(filtered[:10]) + " ..."

            lemma_map = {"handling":"handle","rising":"rise","sparked":"spark",
                         "continues":"continue","depreciate":"depreciate",
                         "prices":"price","elections":"election","outrage":"outrage"}
            lemmatized = [lemma_map.get(tok, tok) for tok in filtered]
            stages["6. Lemmatized"] = str(lemmatized[:10]) + " ..."

            for stage, result in stages.items():
                color = "#1F5A9D" if "6." in stage else "#2563EB" if "0." in stage else "#4B5563"
                st.markdown(f"""
                <div style='background:#FFFFFF; border:1px solid #E5E7EB;
                            border-radius:8px; padding:0.6rem 0.9rem; margin-bottom:0.4rem;'>
                    <span style='font-size:0.72rem; font-weight:600; color:{color};
                                 text-transform:uppercase; letter-spacing:0.05em;'>{stage}</span><br>
                    <code style='font-family:IBM Plex Mono,monospace; font-size:0.8rem;
                                 color:#111827; word-break:break-word;'>{result[:180]}</code>
                </div>""", unsafe_allow_html=True)

        section("Custom Stopwords (Sample)")
        custom_sw_groups = {
            "Generic News": ["said", "stated", "added", "noted", "report", "journalist", "published"],
            "Ghana Filler":  ["ghana", "ghanaian", "accra", "kumasi", "country", "nation"],
            "Election Boilerplate": ["election", "vote", "voter", "ballot", "electoral", "candidate"],
            "Temporal":     ["year", "month", "week", "today", "yesterday", "recent"],
        }
        for group, words in custom_sw_groups.items():
            st.markdown(f"**{group}:** " + " - ".join([f"`{w}`" for w in words]))

# 
# PAGE: TOPIC MODELING
# 

elif page == "Topic Modeling":

    st.markdown('<div class="hero-banner"><div class="flag-stripe"><span style="background:#CE1126;"></span><span style="background:#D4AF37;"></span><span style="background:#006B3F;"></span></div><div class="hero-title">Topic Modeling</div><p class="hero-sub">LDA coherence optimization - BERTopic semantic clustering - 10 election themes discovered</p></div>', unsafe_allow_html=True)

    metric_cards({
        "Optimal k (LDA)": str(BEST_K),
        "Coherence Score": f"{metrics['lda_coherence']:.3f}",
        "Perplexity": f"{metrics['lda_perplexity']:.2f}",
        "BERTopic Topics": str(metrics["bertopic_topics"]),
        "Outlier Rate": f"{metrics['bertopic_outlier_pct']}%",
        "Avg Topic Prob": f"{metrics['avg_topic_probability']:.2f}",
    })

    tabs = st.tabs(["LDA", "BERTopic", "Topic Keywords", "Word Clouds"])

    #  LDA Tab 
    with tabs[0]:
        col1, col2 = st.columns([3, 2])
        with col1:
            section("Coherence Score Optimization")
            k_range = list(COHERENCE_SCORES.keys())
            scores  = list(COHERENCE_SCORES.values())
            st.plotly_chart(fig_coherence(k_range, scores, BEST_K), use_container_width=True)

        with col2:
            section("i How LDA Works")
            info("""
            <b>Latent Dirichlet Allocation (LDA)</b> treats each document as a mixture 
            of topics and each topic as a distribution over words.
            <br><br>
            Think of it like a recipe: each article is a mix of 
            "ingredients" (topics). LDA finds the most likely set 
            of recipes that explain all the articles at once.
            <br><br>
            We tested <b>k = 4 to 12</b> topics and selected the value 
            that maximized <b>c_v coherence</b> - a measure of how 
            semantically related the top words in each topic are.
            <br><br><b>k = 10</b> gave the best score of <b>0.531</b>.
            """)

        section("LDA Hyperparameters")
        params_df = pd.DataFrame([
            {"Parameter": "num_topics",   "Value": "10 (coherence-optimal)", "Rationale": "Best c_v coherence score"},
            {"Parameter": "passes",       "Value": "20",                     "Rationale": "Full corpus iterations for convergence"},
            {"Parameter": "iterations",   "Value": "400",                    "Rationale": "Per-document E-step iterations"},
            {"Parameter": "alpha",        "Value": "auto",                   "Rationale": "Asymmetric Dirichlet prior (auto-tuned)"},
            {"Parameter": "eta",          "Value": "auto",                   "Rationale": "Word distribution prior (auto-tuned)"},
            {"Parameter": "chunksize",    "Value": "100",                    "Rationale": "Mini-batch size for online training"},
            {"Parameter": "random_state", "Value": "42",                     "Rationale": "Reproducibility"},
        ])
        st.dataframe(params_df, use_container_width=True, hide_index=True)

    #  BERTopic Tab 
    with tabs[1]:
        col1, col2 = st.columns([3, 2])
        with col1:
            section("BERTopic Results")
            st.dataframe(
                BERTOPIC_SUMMARY,
                use_container_width=True,
                hide_index=True,
            )

        with col2:
            section("BERTopic vs LDA")
            comparison = [
                ("Representation",   "Bag-of-Words",           "Sentence Embeddings"),
                ("Semantic Nuance",  "Limited",              "Strong"),
                ("Speed",            "Fast",                 "Slower"),
                ("Interpretability", "Probabilistic",        "Cluster-based"),
                ("Outlier Handling", "None",                 "Cluster -1"),
                ("Num Topics",       "Pre-specified (k=10)",   "Auto-discovered"),
            ]
            for feat, lda_val, bert_val in comparison:
                st.markdown(f"""
                <div style='display:grid; grid-template-columns:1fr 1fr 1fr;
                            background:#FFFFFF; border:1px solid #E5E7EB;
                            border-radius:8px; padding:0.5rem 0.8rem; margin-bottom:0.3rem;
                            font-size:0.82rem; gap:0.5rem;'>
                    <span style='color:#4B5563;'>{feat}</span>
                    <span style='color:#60A5FA;'>{lda_val}</span>
                    <span style='color:#34D399;'>{bert_val}</span>
                </div>""", unsafe_allow_html=True)

        section("BERTopic Architecture")
        arch_steps = [
            ("Embed", "all-MiniLM-L6-v2 -> 384-dim sentence vectors per document"),
            ("Reduce", "UMAP dimensionality reduction -> 5-dim manifold"),
            ("Cluster", "HDBSCAN density-based clustering -> natural topic groups"),
            ("Represent", "c-TF-IDF class representation per cluster"),
            ("Label", "Top-10 representative words per cluster"),
        ]
        cols = st.columns(5)
        for i, (label, desc) in enumerate(arch_steps):
            with cols[i]:
                st.markdown(f"""
                <div style='background:#FFFFFF; border:1px solid #E5E7EB;
                            border-radius:10px; padding:0.8rem; text-align:center; height:110px;'>
                    <div style='font-family:Arial,Helvetica,sans-serif; font-size:1rem;
                                font-weight:700; color:#1F5A9D;'>{label}</div>
                    <div style='font-size:0.7rem; color:#4B5563; margin-top:0.4rem;
                                line-height:1.4;'>{desc}</div>
                </div>""", unsafe_allow_html=True)

    #  Keywords Tab 
    with tabs[2]:
        section("Top Keywords per Topic")
        selected_topic = st.selectbox("Select a topic:", TOPIC_LABELS[:n_topics_display])
        col1, col2 = st.columns(2)
        kw_data = TOPIC_KEYWORDS.get(selected_topic, {})

        with col1:
            color_idx = TOPIC_LABELS.index(selected_topic) % len(TOPIC_COLORS)
            st.plotly_chart(
                fig_topic_keywords_bar(kw_data, selected_topic, TOPIC_COLORS[color_idx]),
                use_container_width=True
            )
        with col2:
            st.markdown(f"""
            <div style='background:#FFFFFF; border:1px solid #E5E7EB;
                        border-radius:10px; padding:1.2rem; margin-top:2rem;'>
                <div style='font-family:Arial,Helvetica,sans-serif; font-size:1rem; font-weight:700;
                            color:#1F5A9D; margin-bottom:0.8rem;'>{selected_topic}</div>
                <div style='display:flex; flex-wrap:wrap; gap:0.4rem;'>
            """, unsafe_allow_html=True)

            for word, weight in kw_data.items():
                opacity = int(weight / max(kw_data.values()) * 100)
                st.markdown(f"""
                <span style='background:#EFF6FF;
                             color:#1F5A9D; border:1px solid #BFDBFE; border-radius:6px; padding:0.25rem 0.6rem;
                             font-size:0.8rem; font-weight:600; display:inline-block;
                             margin:0.15rem;'>{word} ({weight:.3f})</span>
                """, unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

    #  Word Clouds Tab 
    with tabs[3]:
        section("Topic Word Clouds")
        cols = st.columns(2)
        for i, topic in enumerate(TOPIC_LABELS[:n_topics_display]):
            with cols[i % 2]:
                img_b64 = generate_wordcloud_b64(TOPIC_KEYWORDS[topic], topic)
                st.markdown(f"""
                <div style='background:#FFFFFF; border:1px solid #E5E7EB;
                            border-radius:10px; overflow:hidden; margin-bottom:1rem;'>
                    <div style='background:#F9FAFB; padding:0.5rem 0.8rem;
                                font-size:0.8rem; font-weight:600; color:#1F5A9D;
                                border-bottom:1px solid #EFF6FF;'>
                        {topic}
                    </div>
                    <img src="data:image/png;base64,{img_b64}"
                         style="width:100%; display:block;" />
                </div>""", unsafe_allow_html=True)

# 
# PAGE: ANALYSIS
# 

elif page == "Analysis":

    st.markdown('<div class="hero-banner"><div class="flag-stripe"><span style="background:#CE1126;"></span><span style="background:#D4AF37;"></span><span style="background:#006B3F;"></span></div><div class="hero-title">Results & Analysis</div><p class="hero-sub">Four research questions - Outlet comparison - Voter priority gap - Party framing</p></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "(i) Dominant Topics",
        "(ii) Outlet Agenda",
        "(iii) Narrow vs Broad",
        "(iv) Party Framing",
        "Voter Gap"
    ])

    #  (i) Dominant Topics 
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            section("Topic Distribution")
            st.plotly_chart(fig_topic_distribution(df), use_container_width=True)
        with col2:
            section("Topic Treemap")
            st.plotly_chart(fig_topic_treemap(df), use_container_width=True)

        section("Key Finding")
        col_a, col_b, col_c = st.columns(3)
        findings = [
            ("#1 - Economy", GHANA_GOLD,
             "Inflation & Economic Hardship leads all outlets. The depreciating cedi, fuel prices, and food costs were ubiquitous. This directly maps to the 62% voter priority for economy in the GIA survey."),
            ("#2 - Corruption", GHANA_RED,
             "Corruption & Governance ranked 2nd in private outlets and 3rd overall. Private online outlets drove this topic, particularly around judgment debt payments and procurement irregularities."),
            ("#3 - Electoral Process", GHANA_GREEN,
             "State-owned outlets (Graphic, Ghanaian Times) elevated Electoral Process coverage - EC operations, biometric verification - above economic issues, revealing their institutional bias."),
        ]
        for (title, color, text), col in zip(findings, [col_a, col_b, col_c]):
            with col:
                st.markdown(f"""
                <div style='background:#FFFFFF; border-top:3px solid {color};
                            border-radius:10px; padding:1rem; height:170px;'>
                    <div style='font-family:Arial,Helvetica,sans-serif; font-weight:700;
                                color:{color}; font-size:0.95rem;'>{title}</div>
                    <div style='font-size:0.8rem; color:#4B5563; margin-top:0.5rem;
                                line-height:1.5;'>{text}</div>
                </div>""", unsafe_allow_html=True)

    #  (ii) Outlet Agenda 
    with tabs[1]:
        section("Topic x Outlet Heatmap")
        st.plotly_chart(fig_outlet_heatmap(df), use_container_width=True)

        section("Coverage Mix by Outlet")
        st.plotly_chart(fig_stacked_bar_outlet(df), use_container_width=True)

        section("Interpretation")
        interp = [
            ("State-Owned", "#CE1126", "Daily Graphic & Ghanaian Times gave 22-24% of coverage to Electoral Process - twice the private outlet average. Economic hardship and corruption received significantly lower weight."),
            ("Private Online", "#3B82F6", "MyJoyOnline & Citinewsroom led on Economy (20-22%), Corruption (16-18%), and Galamsey (13-14%). They were more likely to platform investigative and grievance-based narratives."),
            ("Private Print", "#006B3F", "Daily Guide showed the highest proportion of Corruption coverage (20%) of any outlet - reflecting its editorial positioning as a pro-opposition print outlet."),
        ]
        cols = st.columns(3)
        for (label, color, text), col in zip(interp, cols):
            with col:
                st.markdown(f"""
                <div style='background:#FFFFFF; border-left:3px solid {color};
                            border-radius:0 10px 10px 0; padding:1rem; height:150px;'>
                    <div style='font-weight:700; color:{color}; font-size:0.9rem;'>{label}</div>
                    <div style='font-size:0.79rem; color:#4B5563; margin-top:0.4rem;
                                line-height:1.5;'>{text}</div>
                </div>""", unsafe_allow_html=True)

    #  (iii) Narrow vs Broad 
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            section("Overall Split")
            st.plotly_chart(fig_narrow_broad_donut(df), use_container_width=True)
        with col2:
            section("By Outlet")
            st.plotly_chart(fig_narrow_broad_by_outlet(df), use_container_width=True)

        section("Framework Definition")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div style='background:rgba(212,175,55,0.06); border:1px solid #E5E7EB;
                        border-radius:10px; padding:1.1rem;'>
                <div style='font-family:Arial,Helvetica,sans-serif; font-weight:700; color:#1F5A9D;
                            font-size:0.95rem; margin-bottom:0.6rem;'>
                    Narrow Coverage
                </div>
                <div style='font-size:0.82rem; color:#374151; line-height:1.6;'>
                    Focuses on the <b>electoral contest itself</b> - who's winning, 
                    rally attendance, party tactics, EC logistics, debate performances.<br><br>
                    Topics: NPP Campaign - NDC Campaign - Electoral Process - 
                    Security - Parliamentary Races
                </div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div style='background:rgba(0,107,63,0.06); border:1px solid rgba(0,107,63,0.3);
                        border-radius:10px; padding:1.1rem;'>
                <div style='font-family:Arial,Helvetica,sans-serif; font-weight:700; color:#34D399;
                            font-size:0.95rem; margin-bottom:0.6rem;'>
                    Broad Coverage
                </div>
                <div style='font-size:0.82rem; color:#374151; line-height:1.6;'>
                    Focuses on <b>policy issues and structural conditions</b> - 
                    economic hardship, environmental damage, corruption, 
                    education quality, youth unemployment.<br><br>
                    Topics: Economy - Galamsey - Corruption - Free SHS - Jobs
                </div>
            </div>""", unsafe_allow_html=True)

        success("<b>Finding:</b> Ghana's 2024 election media was predominantly <b>Broad (issue-based)</b> rather than narrow horse-race journalism - a positive indicator for the quality of public discourse. However, state outlets skewed more Narrow via their heavy Electoral Process coverage.")

    #  (iv) Party Framing 
    with tabs[3]:
        section("NPP vs. NDC Topic Focus")
        st.plotly_chart(fig_party_framing(df), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style='background:rgba(206,17,38,0.08); border:1px solid rgba(206,17,38,0.3);
                        border-radius:10px; padding:1.1rem;'>
                <div style='font-family:Arial,Helvetica,sans-serif; font-weight:700; color:#F87171;
                            font-size:0.95rem; margin-bottom:0.6rem;'>NPP Narrative</div>
                <div style='font-size:0.82rem; color:#374151; line-height:1.6;'>
                    NPP-tagged articles emphasized <b>digital achievements</b> (mobile money, 
                    Ghana.gov), <b>Free SHS expansion</b>, and <b>infrastructure projects</b> 
                    (roads, dams). This was an incumbency-defence posture: "look at what we built."<br><br>
                    Weakness: Economy and corruption topics appeared in NPP coverage too - 
                    opponents used NPP press releases as hooks to pivot to criticism.
                </div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style='background:rgba(0,107,63,0.08); border:1px solid rgba(0,107,63,0.3);
                        border-radius:10px; padding:1.1rem;'>
                <div style='font-family:Arial,Helvetica,sans-serif; font-weight:700; color:#34D399;
                            font-size:0.95rem; margin-bottom:0.6rem;'>NDC Narrative</div>
                <div style='font-size:0.82rem; color:#374151; line-height:1.6;'>
                    NDC-tagged articles concentrated on <b>economic pain</b>, 
                    <b>corruption allegations</b>, and <b>galamsey failure</b> - 
                    Mahama's "Reset" message dominated. This anti-incumbency frame 
                    resonated because it matched lived experience.<br><br>
                    Strength: The NDC narrative aligned perfectly with the top 3 
                    voter concerns in the GIA pre-election survey.
                </div>
            </div>""", unsafe_allow_html=True)

    #  Voter Gap 
    with tabs[4]:
        section("Media Agenda vs. Voter Priorities")
        st.plotly_chart(fig_media_voter_gap(df), use_container_width=True)

        warn("""<b>Key Gap:</b><b>Unemployment/Jobs</b> was the second-highest voter priority (48%) 
        but received only modest media coverage. This represents the most significant 
        agenda-setting failure - millions of unemployed Ghanaians felt their core 
        concern was under-represented in the election narrative.""")

# 
# PAGE: AI AGENT
# 

elif page == "AI Agent - Prof. Kwame":

    #  Session state init 
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_section" not in st.session_state:
        st.session_state.agent_section = "overview"

    #  Header 
    st.markdown("""
    <div class="hero-banner">
        <div class="flag-stripe">
            <span style="background:#CE1126;"></span>
            <span style="background:#D4AF37;"></span>
            <span style="background:#006B3F;"></span>
        </div>
        <div class="hero-title">Prof. Kwame - AI Research Agent</div>
        <p class="hero-sub">
            Live AI agent powered by Gemini - Ask anything about the project, 
            methodology, findings, or Ghana's 2024 election
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_chat, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        section("Quick Explain")
        st.markdown("<div style='font-size:0.8rem; color:#4B5563; margin-bottom:0.5rem;'>Ask Prof. Kwame about:</div>", unsafe_allow_html=True)

        explain_options = {
            "Project Overview": "overview",
            "Data Collection":  "data",
            "Preprocessing":    "preprocessing",
            "LDA Model":        "lda",
            "Dominant Topics":  "topics",
            "Outlet Agenda":    "outlets",
            "Narrow vs Broad":  "narrow_broad",
            "Party Framing":  "party",
            "Conclusion":       "conclusion",
        }

        for label, section_key in explain_options.items():
            if st.button(label, use_container_width=True, key=f"btn_{section_key}"):
                prompt = get_contextual_prompt(section_key)
                st.session_state.agent_section = section_key
                st.session_state._pending_prompt = prompt

        st.markdown("---")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("---")
        section("Suggested Questions")
        for q in SUGGESTED_QUESTIONS[:5]:
            if st.button(q, use_container_width=True, key=f"sq_{q[:20]}"):
                st.session_state._pending_prompt = q

    with col_chat:
        #  Agent header card 
        st.markdown("""
        <div class="agent-header">
            <div class="agent-avatar"></div>
            <div>
                <div class="agent-name">Professor Kwame</div>
                <div class="agent-status">Online - NLP & Ghanaian Political Science Expert</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        #  Chat messages 
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""
                <div class="chat-bubble-agent"><b>Akwaaba!</b>  I'm Professor Kwame, your AI guide to this Ghana 2024 
                    election media analysis.<br><br>
                    I can explain how we collected and processed the data, walk you through 
                    LDA and BERTopic topic modeling, interpret our findings on media bias, 
                    narrow vs. broad coverage, party framing - or discuss why Mahama won 
                    and what the media's agenda had to do with it.<br><br>
                    What would you like to explore? Use the quick buttons on the right, 
                    try a suggested question, or ask me anything! 
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-bubble-agent">{msg["content"]}</div>', unsafe_allow_html=True)

        #  Suggested chips (shown when empty) 
        if not st.session_state.chat_history:
            st.markdown("<div style='margin-top:0.5rem;'>", unsafe_allow_html=True)
            for q in SUGGESTED_QUESTIONS[5:]:
                st.markdown(f'<span class="chip">{q}</span>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        #  Input 
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

        with st.form(key="chat_form", clear_on_submit=True):
            cols = st.columns([5, 1])
            with cols[0]:
                user_input = st.text_input(
                    "Your message",
                    placeholder="Ask Prof. Kwame anything about the project...",
                    label_visibility="collapsed",
                )
            with cols[1]:
                submitted = st.form_submit_button("Send", use_container_width=True)

        #  Handle pending prompt from buttons 
        pending = st.session_state.pop("_pending_prompt", None)
        final_input = pending or (user_input if submitted and user_input.strip() else None)

        if final_input:
            st.session_state.chat_history.append({
                "role": "user", "content": final_input
            })

            st.markdown(f'<div class="chat-bubble-user">{final_input}</div>', unsafe_allow_html=True)

            # Stream response
            response_placeholder = st.empty()
            full_response = ""
            with st.spinner(""):
                try:
                    for chunk in stream_agent_response(
                        final_input, st.session_state.chat_history[:-1]
                    ):
                        full_response += chunk
                        response_placeholder.markdown(
                            f'<div class="chat-bubble-agent">{full_response}</div>',
                            unsafe_allow_html=True
                        )
                    response_placeholder.markdown(
                        f'<div class="chat-bubble-agent">{full_response}</div>',
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    full_response = (
                        "I'm having trouble connecting to Professor Kwame right now. Please ensure your "
                        "GEMINI_API_KEY secret environment variable is set correctly. "
                        f"Error: {str(e)[:120]}"
                    )
                    response_placeholder.markdown(
                        f'<div class=\"chat-bubble-agent\" style=\"border-color:rgba(206,17,38,0.4);\">⚠️ {full_response}</div>',
                        unsafe_allow_html=True
                    )

            st.session_state.chat_history.append({
                "role": "assistant", "content": full_response
            })
            st.rerun()
