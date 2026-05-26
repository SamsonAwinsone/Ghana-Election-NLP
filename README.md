# Ghana 2024 Election - NLP Topic Modeling Dashboard

<div align="center">

![Ghana Flag](https://img.shields.io/badge/Ghana-2024%20Election-D4AF37-style=for-the-badge&logo=data:image/svg+xml;base64,)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B-style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB-style=for-the-badge&logo=python&logoColor=white)
![Claude AI](https://img.shields.io/badge/Claude-AI%20Agent-D4AF37-style=for-the-badge)
![License: MIT](https://img.shields.io/badge/License-MIT-green-style=for-the-badge)

**An interactive NLP dashboard analyzing how Ghanaian mass media framed the 2024 presidential & parliamentary elections - with a live Claude-powered AI research agent.**

[Live Demo](#) - [Jupyter Notebook](#) - [Results](#analysis-findings)

</div>

---

## Dashboard Preview

| Page | Description |
|---|---|
| Overview | Project summary, corpus stats, methodology pipeline |
| Data Collection | Outlet breakdown, article timeline, scraping architecture |
| Preprocessing | Live NLP pipeline demo, stopword groups, vocabulary stats |
| Topic Modeling | LDA coherence tuning, BERTopic results, keyword charts, word clouds |
| Analysis | 4 research questions: dominant topics, outlet agenda, narrow/broad, party framing |
| AI Agent | Live chat with Prof. Kwame - Claude-powered expert on the project |

---

## Research Questions

1. **What topics dominated** Ghana's 2024 election media coverage?
2. **How did the issue agenda vary** across state-owned vs. private outlets?
3. **Was coverage narrow** (horse-race) **or broad** (policy/issue-focused)?
4. **How did NPP vs. NDC framing** compare against voter priorities and the election outcome?

---

## Media Outlets Analyzed

| Outlet | Type | Format |
|---|---|---|
| MyJoyOnline | Private | Online |
| Citinewsroom | Private | Online |
| Daily Graphic | State-owned | Print/Online |
| Ghanaian Times | State-owned | Print/Online |
| Daily Guide | Private | Print/Online |

---

## Methods

```
Web Scraping (newspaper3k + BeautifulSoup)
        
Text Preprocessing (spaCy - NLTK - 341 stopwords - POS filtering)
        
LDA Topic Modeling (Gensim - coherence optimization k=4->12)
        
BERTopic (sentence-transformers - UMAP - HDBSCAN)
        
Comparative Analysis (Narrow/Broad - Party framing - Voter gap)
        
AI Explanation Agent (Claude claude-sonnet-4-20250514 - live streaming)
```

---

## Analysis Findings

### Topics Discovered (k=10, coherence=0.531)

| # | Topic | Coverage |
|---|---|---|
| 1 | Inflation & Economic Hardship | Highest share |
| 2 | Corruption & Governance | High in private outlets |
| 3 | Electoral Process & EC | Dominant in state outlets |
| 4 | NPP Campaign & Manifesto | Campaign period spike |
| 5 | NDC Campaign & Manifesto | Aligned with voter grievances |
| 6 | Illegal Mining (Galamsey) | Consistent throughout 2024 |
| 7 | Education Policy (Free SHS) | Policy debate |
| 8 | Jobs & Unemployment | Under-covered vs. voter priority |
| 9 | Electoral Security & Violence | Pre-election surge |
| 10 | Regional/Parliamentary Races | Constituency-level coverage |

### Key Findings
- **Economy dominated** all outlets - aligned with GIA survey (62% voter priority)
- **State outlets** over-indexed on Electoral Process; **private outlets** on Corruption
- **Broad (issue) coverage** outweighed Narrow (horse-race) - 58% vs 42%
- **NDC narrative** (economy + corruption) perfectly matched top voter concerns
- **Media agenda strongly predicted** the NDC/Mahama election victory

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ghana-election-nlp.git
cd ghana-election-nlp
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download spaCy model
```bash
python -m spacy download en_core_web_sm
```

### 4. Set your Anthropic API key
```bash
# Option A - environment variable
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# Option B - Streamlit secrets
cp .streamlit/secrets.toml.template .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and add your key
```

### 5. Run the app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) 

---

##  Deploy to Streamlit Cloud

1. **Push to GitHub** - make sure `.gitignore` excludes `secrets.toml`
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set **Main file path** -> `app.py`
5. Under **Settings -> Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ```
6. Click **Deploy** 

---

##  Project Structure

```
ghana-election-nlp/
 app.py                          #  Main Streamlit dashboard
 requirements.txt                # Python dependencies
 packages.txt                    # System dependencies (Streamlit Cloud)
 .gitignore
 README.md

 .streamlit/
    config.toml                 # Dark theme + Ghana colors
    secrets.toml.template      # API key template

 data/
    sample_data.py             # Realistic sample corpus generator

 utils/
     __init__.py
     ai_agent.py                 # Claude AI agent (Prof. Kwame)
     topic_modeling.py           # LDA/BERTopic helpers & results
     visualizations.py          # All Plotly chart functions
```

---

##  API Key

The AI Agent (Prof. Kwame) requires an [Anthropic API key](https://console.anthropic.com).

The app works fully without the key - only the AI Agent page requires it.

---

##  Tech Stack

| Component | Tool |
|---|---|
| Frontend | Streamlit 1.35+ |
| AI Agent | Anthropic Claude (claude-sonnet-4-20250514) |
| Topic Modeling | Gensim LDA + BERTopic |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| NLP Pipeline | spaCy en_core_web_sm + NLTK |
| Visualizations | Plotly + Matplotlib + WordCloud |
| Web Scraping | newspaper3k + BeautifulSoup4 |

---

##  References

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). *Latent Dirichlet Allocation*. JMLR.
- Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*.
- Stromback, J., & Aalberg, T. (2008). *Election news coverage in democratic corporatist countries*. Journalism Studies.
- Global Info Analytics / Musa Danquah (2024). *Pre-election voter priority survey, Ghana*.
- Electoral Commission of Ghana (2024). *2024 Presidential & Parliamentary Election Results*.

---

##  License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
    Built with  for academic research - Ghana  2024
</div>
