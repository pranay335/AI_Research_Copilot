import feedparser
from urllib.parse import quote
from langchain_core.tools import tool


@tool
def search_news(query: str) -> str:
    """Search Google News RSS for recent news about a topic. 
    Returns article titles, publication dates, and source names.
    Do not open or fetch the article URLs."""

    encoded_query = quote(query)

    url = (
        "https://news.google.com/rss/search?"
        f"q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    )

    feed = feedparser.parse(url)

    if not feed.entries:
        return "No news articles found."

    results = []

    for article in feed.entries[:5]:
        title = article.get("title", "No title")
        published = article.get("published", "Unknown date")

        source = "Unknown source"

        if hasattr(article, "source") and article.source:
            source = article.source.get("title", "Unknown source")

        results.append(
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Published: {published}"
        )

    return "\n\n---\n\n".join(results)