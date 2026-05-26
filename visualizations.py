"""
utils/visualizations.py
All Plotly chart functions for the Streamlit dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io, base64

# Ghana-themed color palette
GHANA_GOLD    = "#D4AF37"
GHANA_RED     = "#CE1126"
GHANA_GREEN   = "#006B3F"
DARK_BG       = "#F9FAFB"
CARD_BG       = "#FFFFFF"
ACCENT        = "#1F5A9D"

TOPIC_COLORS = [
    "#D4AF37","#CE1126","#006B3F","#3B82F6","#F97316",
    "#8B5CF6","#EC4899","#14B8A6","#F59E0B","#6366F1",
]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(family="Arial, Helvetica, sans-serif", color="#111827"),
)


# 1. Article Volume Charts
# 

def fig_article_counts(df: pd.DataFrame):
    counts = df["outlet"].value_counts().reset_index()
    counts.columns = ["Outlet", "Articles"]
    counts["Outlet"] = counts["Outlet"].str.replace("_", " ")

    fig = px.bar(
        counts, x="Outlet", y="Articles",
        color="Articles",
        color_continuous_scale=[[0, "#1E3A5F"], [0.5, GHANA_GOLD], [1, GHANA_RED]],
        text="Articles",
    )
    fig.update_traces(textposition="outside", textfont_size=13)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Articles Collected per Outlet", font_size=16),
        xaxis_title="", yaxis_title="Article Count",
        coloraxis_showscale=False,
    )
    return fig


def fig_timeline(df: pd.DataFrame):
    ts = df.copy()
    ts["month"] = pd.to_datetime(ts["date"]).dt.to_period("M").dt.to_timestamp()
    monthly = ts.groupby(["month", "outlet"])["headline"].count().reset_index()
    monthly.columns = ["Month", "Outlet", "Articles"]
    monthly["Outlet"] = monthly["Outlet"].str.replace("_", " ")

    fig = px.line(
        monthly, x="Month", y="Articles", color="Outlet",
        markers=True, color_discrete_sequence=TOPIC_COLORS,
    )
    election_day = pd.Timestamp("2024-12-07")
    fig.add_shape(
        type="line",
        x0=election_day,
        x1=election_day,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color=GHANA_RED, dash="dash", width=2),
    )
    fig.add_annotation(
        x=election_day,
        y=1,
        xref="x",
        yref="paper",
        text="Election Day",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(color=GHANA_RED),
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Coverage Volume Over Time", font_size=16),
        xaxis_title="", yaxis_title="Articles Published",
    )
    return fig


# 2. Topic Distribution
# 

def fig_topic_distribution(df: pd.DataFrame):
    topic_counts = df["lda_topic_label"].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Count"]
    topic_counts["Pct"] = (topic_counts["Count"] / len(df) * 100).round(1)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=topic_counts["Topic"],
        x=topic_counts["Pct"],
        orientation="h",
        marker=dict(
            color=topic_counts["Pct"],
            colorscale=[[0, "#1E3A5F"], [0.5, GHANA_GOLD], [1, GHANA_RED]],
            line=dict(color="rgba(255,255,255,0.05)", width=1),
        ),
        text=[f"{p}%" for p in topic_counts["Pct"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:.1f}% of all articles<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Dominant Topics  -  All Outlets", font_size=16),
        xaxis_title="Share of Coverage (%)",
        yaxis=dict(categoryorder="total ascending"),
        height=420,
    )
    return fig


def fig_topic_treemap(df: pd.DataFrame):
    topic_counts = df["lda_topic_label"].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Count"]

    fig = px.treemap(
        topic_counts, path=["Topic"], values="Count",
        color="Count",
        color_continuous_scale=[[0, "#1E3A5F"], [0.4, GHANA_GREEN], [0.7, GHANA_GOLD], [1, GHANA_RED]],
    )
    fig.update_traces(
        textinfo="label+percent entry",
        hovertemplate="<b>%{label}</b><br>%{value} articles<br>%{percentRoot:.1%}<extra></extra>",
    )
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text="Topic Treemap", font_size=16), height=380)
    return fig


# 3. Outlet Comparison
# 

def fig_outlet_heatmap(df: pd.DataFrame):
    pivot = (
        df.groupby(["outlet", "lda_topic_label"])
        .size()
        .reset_index(name="count")
    )
    totals = df.groupby("outlet").size().reset_index(name="total")
    pivot = pivot.merge(totals, on="outlet")
    pivot["pct"] = (pivot["count"] / pivot["total"] * 100).round(1)
    matrix = pivot.pivot(index="outlet", columns="lda_topic_label", values="pct").fillna(0)
    matrix.index = matrix.index.str.replace("_", " ")

    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=matrix.columns.tolist(),
        y=matrix.index.tolist(),
        colorscale=[[0, DARK_BG], [0.5, GHANA_GOLD], [1, GHANA_RED]],
        text=matrix.values.round(1),
        texttemplate="%{text}%",
        hovertemplate="<b>%{y}</b> -> <b>%{x}</b><br>%{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Topic x Outlet Heatmap (% of each outlet's coverage)", font_size=16),
        xaxis=dict(tickangle=-30),
        height=350,
    )
    return fig


def fig_stacked_bar_outlet(df: pd.DataFrame):
    pivot = (
        df.groupby(["outlet", "lda_topic_label"])
        .size()
        .reset_index(name="count")
    )
    totals = df.groupby("outlet").size().reset_index(name="total")
    pivot = pivot.merge(totals, on="outlet")
    pivot["pct"] = (pivot["count"] / pivot["total"] * 100).round(1)
    pivot["outlet"] = pivot["outlet"].str.replace("_", " ")

    topics = df["lda_topic_label"].value_counts().index.tolist()
    fig = go.Figure()
    for i, topic in enumerate(topics):
        sub = pivot[pivot["lda_topic_label"] == topic]
        fig.add_trace(go.Bar(
            name=topic,
            x=sub["outlet"],
            y=sub["pct"],
            marker_color=TOPIC_COLORS[i % len(TOPIC_COLORS)],
            hovertemplate=f"<b>{topic}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="stack",
        title=dict(text="Coverage Mix by Outlet", font_size=16),
        yaxis_title="Share (%)",
        legend=dict(orientation="v", x=1.01, y=1),
        height=420,
    )
    return fig


# 4. Narrow vs Broad
# 

def fig_narrow_broad_donut(df: pd.DataFrame):
    counts = df["coverage_type"].value_counts()
    colors = {"Broad": GHANA_GREEN, "Narrow": GHANA_GOLD, "Unclassified": "#6B7280"}

    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.6,
        marker=dict(colors=[colors.get(l, ACCENT) for l in counts.index]),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value} articles (%{percent})<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{len(df)}</b><br>Articles",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="#111827"),
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Narrow vs. Broad Coverage  -  All Outlets", font_size=16),
        height=360,
    )
    return fig


def fig_narrow_broad_by_outlet(df: pd.DataFrame):
    ratio = (
        df.groupby(["outlet", "coverage_type"])
        .size()
        .unstack(fill_value=0)
    )
    ratio_pct = ratio.div(ratio.sum(axis=1), axis=0) * 100
    ratio_pct.index = ratio_pct.index.str.replace("_", " ")

    fig = go.Figure()
    color_map = {"Broad": GHANA_GREEN, "Narrow": GHANA_GOLD, "Unclassified": "#6B7280"}
    for col in ratio_pct.columns:
        fig.add_trace(go.Bar(
            name=col,
            x=ratio_pct.index,
            y=ratio_pct[col].round(1),
            marker_color=color_map.get(col, ACCENT),
            text=ratio_pct[col].round(1).astype(str) + "%",
            textposition="inside",
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="stack",
        title=dict(text="Narrow vs. Broad Coverage by Outlet", font_size=16),
        yaxis_title="%",
        height=360,
    )
    return fig


# 5. Party Framing
# 

def fig_party_framing(df: pd.DataFrame):
    party_df = df[df["party_frame"].isin(["NPP", "NDC"])].copy()
    pivot = (
        party_df.groupby(["party_frame", "lda_topic_label"])
        .size()
        .reset_index(name="count")
    )
    totals = party_df.groupby("party_frame").size().reset_index(name="total")
    pivot = pivot.merge(totals, on="party_frame")
    pivot["pct"] = (pivot["count"] / pivot["total"] * 100).round(1)

    fig = px.bar(
        pivot, x="lda_topic_label", y="pct", color="party_frame",
        barmode="group",
        color_discrete_map={"NPP": GHANA_RED, "NDC": GHANA_GREEN},
        labels={"pct": "Share (%)", "lda_topic_label": "", "party_frame": "Party"},
        text="pct",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="NPP vs. NDC  -  Topic Focus in Media Coverage", font_size=16),
        xaxis=dict(tickangle=-30),
        height=420,
    )
    return fig


# 6. Media vs Voter Gap Analysis
# 

def fig_media_voter_gap(df: pd.DataFrame):
    voter_data = pd.DataFrame([
        {"Issue": "Economy/Cost of Living",  "Voter_Priority": 62, "Topic": "Inflation & Economic Hardship"},
        {"Issue": "Unemployment/Jobs",        "Voter_Priority": 48, "Topic": "Jobs & Unemployment"},
        {"Issue": "Corruption",               "Voter_Priority": 41, "Topic": "Corruption & Governance"},
        {"Issue": "Galamsey",                 "Voter_Priority": 35, "Topic": "Illegal Mining (Galamsey)"},
        {"Issue": "Education",                "Voter_Priority": 28, "Topic": "Education Policy (Free SHS)"},
        {"Issue": "Security",                 "Voter_Priority": 22, "Topic": "Electoral Security & Violence"},
    ])
    media_share = (df["lda_topic_label"].value_counts(normalize=True) * 100).reset_index()
    media_share.columns = ["Topic", "Media_Share"]
    voter_data = voter_data.merge(media_share, on="Topic", how="left").fillna(0)
    voter_data["Gap"] = (voter_data["Media_Share"] - voter_data["Voter_Priority"]).round(1)

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Voter Priority vs. Media Coverage (%)",
        "Coverage Gap (Media% - Voter%)"
    ))

    fig.add_trace(go.Bar(
        name="Voter Priority", x=voter_data["Issue"], y=voter_data["Voter_Priority"],
        marker_color=GHANA_GOLD, text=voter_data["Voter_Priority"].astype(str)+"%",
        textposition="outside",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        name="Media Coverage", x=voter_data["Issue"], y=voter_data["Media_Share"].round(1),
        marker_color=ACCENT, text=voter_data["Media_Share"].round(1).astype(str)+"%",
        textposition="outside",
    ), row=1, col=1)

    gap_colors = [GHANA_GREEN if g >= 0 else GHANA_RED for g in voter_data["Gap"]]
    fig.add_trace(go.Bar(
        name="Gap", x=voter_data["Issue"], y=voter_data["Gap"],
        marker_color=gap_colors,
        text=[f"{g:+.1f}pp" for g in voter_data["Gap"]],
        textposition="outside",
        showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color="#6B7280", row=1, col=2)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Media Agenda vs. Voter Priorities (Global Info Analytics, 2024)", font_size=16),
        height=420, barmode="group",
    )
    fig.update_xaxes(tickangle=-25)
    return fig


# 7. Word Cloud
# 

def generate_wordcloud_b64(keywords: dict, topic_name: str) -> str:
    """Generate a word cloud from keyword dict and return as base64 PNG."""
    cmap_colors = ["#D4AF37", "#CE1126", "#F97316", "#FBBF24", "#FDE68A"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("ghana", cmap_colors)

    wc = WordCloud(
        width=700, height=320,
        background_color="#FFFFFF",
        colormap=custom_cmap,
        max_words=40,
        prefer_horizontal=0.85,
        collocations=False,
    ).generate_from_frequencies(keywords)

    fig_wc, ax = plt.subplots(figsize=(7, 3.2), facecolor="#FFFFFF")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_facecolor("#FFFFFF")
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# 8. Coherence Score Chart
# 

def fig_coherence(k_range, scores, best_k):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(k_range), y=scores,
        mode="lines+markers",
        line=dict(color=GHANA_GOLD, width=3),
        marker=dict(size=9, color=GHANA_GOLD, symbol="circle"),
        name="Coherence (c_v)",
        hovertemplate="k=%{x}<br>Score=%{y:.4f}<extra></extra>",
    ))
    fig.add_vline(
        x=best_k, line_dash="dash", line_color=GHANA_RED,
        annotation_text=f"Best k={best_k}", annotation_position="top right",
        annotation_font_color=GHANA_RED,
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="LDA Coherence Score by Number of Topics", font_size=16),
        xaxis_title="Number of Topics (k)",
        yaxis_title="Coherence Score (c_v)",
        height=340,
    )
    return fig


# 9. Topic Keywords Bar
# 

def fig_topic_keywords_bar(keywords: dict, topic_name: str, color: str = GHANA_GOLD):
    words = list(keywords.keys())[:10]
    weights = list(keywords.values())[:10]

    fig = go.Figure(go.Bar(
        y=words[::-1], x=weights[::-1],
        orientation="h",
        marker=dict(
            color=weights[::-1],
            colorscale=[[0, "#1E3A5F"], [1, color]],
            line=dict(color="rgba(255,255,255,0.05)", width=1),
        ),
        hovertemplate="<b>%{y}</b>: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=topic_name, font_size=14),
        xaxis_title="Weight",
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
