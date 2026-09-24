import sys
import json
from pathlib import Path

# -------------------------------------------------------
# Path setup: add backend/ to sys.path so all imports work
# -------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import streamlit as st

from intelligence.research_engine import (
    search_articles,
    get_multi_article_topics,
    get_related_articles,
    analyze_research_trends,
)
from intelligence.article_analyzer import analyze_article
from memory.memory import initialize_database, get_research_history, save_research
from agent.agent import graph as agent_graph

# Initialize database on startup
initialize_database()

# -------------------------------------------------------
# Page config
# -------------------------------------------------------
st.set_page_config(
    page_title="AI Research Copilot",
    layout="wide"
)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
st.sidebar.title("About")
st.sidebar.markdown(
    """
- NLP-based research assistant
- Uses LangGraph for agentic workflow
- Uses Groq LLM
- Uses TF-IDF / NMF and other NLP techniques
- Uses SQLite for research memory
    """
)

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Topics (NMF)")
corpus_topics = get_multi_article_topics(num_topics=3, num_words=5)
for topic in corpus_topics:
    kw_str = ", ".join(topic["keywords"])
    st.sidebar.markdown(f"**Topic {topic['topic_id']}:** {kw_str}")

# -------------------------------------------------------
# Main header
# -------------------------------------------------------
st.title("AI Research Copilot")
st.caption("An NLP-based personal assistant for personalized news and technology intelligence.")
st.markdown("---")

# -------------------------------------------------------
# Session state
# -------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# -------------------------------------------------------
# Search input
# -------------------------------------------------------
query = st.text_input(
    "Ask a research question",
    placeholder="e.g. What are AI agents used for in enterprise software?"
)

if st.button("Research"):
    if not query.strip():
        st.warning("Please enter a research question before clicking Research.")
    else:
        with st.spinner("Running NLP pipeline..."):
            try:
                # -----------------------------------------------
                # 1. TF-IDF search — find top matching articles
                # -----------------------------------------------
                matched_articles = search_articles(query, top_k=5)

                # -----------------------------------------------
                # 2. Full NLP analysis on the top article
                # -----------------------------------------------
                top_article_analysis = None
                if matched_articles and matched_articles[0]["similarity"] > 0:
                    top_article_analysis = analyze_article(matched_articles[0])

                # -----------------------------------------------
                # 3. Related articles for the top matched article
                # -----------------------------------------------
                related = []
                if matched_articles and matched_articles[0]["similarity"] > 0:
                    related = get_related_articles(
                        target_article_id=matched_articles[0]["id"],
                        top_k=3
                    )

                # -----------------------------------------------
                # 4. Research trends
                # -----------------------------------------------
                trends = analyze_research_trends()

                # -----------------------------------------------
                # 5. Save this query to memory (compact reference)
                # -----------------------------------------------
                topic_labels = []
                if top_article_analysis:
                    topic_labels = [
                        " & ".join(t["keywords"])
                        for t in top_article_analysis.get("topics", [])
                    ]
                article_refs = [
                    {"id": a["id"], "title": a["title"]}
                    for a in matched_articles[:3]
                    if a["similarity"] > 0
                ]
                save_research(query, topic_labels, article_refs)

                # -----------------------------------------------
                # 6. Get LLM answer using the existing agent
                # -----------------------------------------------
                agent_answer = ""
                try:
                    agent_response = agent_graph.invoke(
                        {"messages": [("user", query)]}
                    )
                    for msg in agent_response.get("messages", []):
                        if getattr(msg, "content", None):
                            agent_answer = msg.content
                except Exception:
                    agent_answer = ""

                # Store everything in session state
                st.session_state.last_query = query
                st.session_state.results = {
                    "matched_articles": matched_articles,
                    "top_article_analysis": top_article_analysis,
                    "related": related,
                    "trends": trends,
                    "agent_answer": agent_answer,
                }

            except Exception as e:
                st.error(f"Error running NLP pipeline: {e}")

# -------------------------------------------------------
# Display results
# -------------------------------------------------------
if st.session_state.results:
    r = st.session_state.results
    matched_articles = r["matched_articles"]
    top_article_analysis = r["top_article_analysis"]
    related = r["related"]
    trends = r["trends"]
    agent_answer = r.get("agent_answer", "")

    st.markdown(f"### Results for: *{st.session_state.last_query}*")
    st.markdown("---")

    # -----------------------------------------------
    # Section 1: Matched Articles (TF-IDF Similarity)
    # -----------------------------------------------
    st.subheader("Matched Articles (TF-IDF Cosine Similarity)")
    if matched_articles:
        for art in matched_articles:
            if art["similarity"] > 0:
                with st.expander(f"{art['title']}  —  similarity: {art['similarity']:.4f}"):
                    col1, col2 = st.columns(2)
                    col1.markdown(f"**Source:** {art['source']}")
                    col2.markdown(f"**Date:** {art['date']}")
                    st.markdown(art["content"])
    else:
        st.info("No articles matched your query.")

    # -----------------------------------------------
    # Section 2: NLP Analysis of Top Article
    # -----------------------------------------------
    if top_article_analysis:
        st.markdown("---")
        st.subheader(f"NLP Analysis — *{top_article_analysis['title']}*")

        col_kw, col_phrases = st.columns(2)

        with col_kw:
            st.markdown("**Top Keywords (TF)**")
            kw_data = {
                kw["word"]: kw["frequency"]
                for kw in top_article_analysis.get("keywords", [])
            }
            if kw_data:
                st.bar_chart(kw_data)

        with col_phrases:
            st.markdown("**Key N-Gram Phrases**")
            phrases = top_article_analysis.get("phrases", [])
            if phrases:
                for p in phrases[:8]:
                    st.markdown(f"- `{p['phrase']}` × {p['frequency']}")
            else:
                st.info("No phrases found.")

        # Topics from NMF
        st.markdown("**Extracted Topics (NMF/TF-IDF)**")
        topics = top_article_analysis.get("topics", [])
        if topics:
            cols = st.columns(len(topics))
            for i, topic in enumerate(topics):
                with cols[i]:
                    st.markdown(f"**Topic {topic['topic_id']}**")
                    for kw in topic["keywords"]:
                        st.markdown(f"- {kw}")
        else:
            st.info("No topics extracted.")

        # Named Entities
        entities = top_article_analysis.get("entities", [])
        if entities:
            st.markdown("**Named Entities (NER)**")
            st.write(entities)

    else:
        st.info("No article scored high enough for deep NLP analysis.")

    # -----------------------------------------------
    # Section 3: Related Articles
    # -----------------------------------------------
    st.markdown("---")
    st.subheader("Related Articles (Cosine Similarity)")
    if related:
        for rel in related:
            st.markdown(
                f"- **{rel['title']}** — similarity: `{rel['similarity']:.4f}`"
            )
    else:
        st.info("No related articles found.")

    # -----------------------------------------------
    # Section 4: Research Trends (NLP-based)
    # -----------------------------------------------
    st.markdown("---")
    st.subheader("Research Trends (NLP Keyword Analysis)")
    if trends:
        for period, data in trends.items():
            st.markdown(f"**Period: {period}**")
            trend_data = {
                t["topic"]: t["mentions"]
                for t in data.get("topics", [])
            }
            if trend_data:
                st.bar_chart(trend_data)
    else:
        st.info("No trend data available.")

    # -----------------------------------------------
    # Section 5: Research History (SQLite Memory)
    # -----------------------------------------------
    st.markdown("---")
    st.subheader("Research History (SQLite Memory)")
    history = get_research_history(limit=5)
    if history:
        for record in history:
            with st.expander(f"{record['created_at'][:19]}  —  {record['query']}"):
                st.markdown(f"**Topics:** {', '.join(record['topics']) if record['topics'] else 'N/A'}")
                if record["articles"]:
                    for a in record["articles"]:
                        st.markdown(f"- Article {a['id']}: {a['title']}")
    else:
        st.info("No research history yet.")

    # -----------------------------------------------
    # Section 6: Research Answer (LLM via LangGraph)
    # -----------------------------------------------
    st.markdown("---")
    st.subheader("Research Answer (LangGraph Agent)")
    if agent_answer:
        st.markdown(agent_answer)
    else:
        st.info("No agent answer available.")
