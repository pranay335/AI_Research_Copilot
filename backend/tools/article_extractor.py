import requests
import trafilatura
from langchain_core.tools import tool


@tool
def extract_article(url: str) -> str:
    """Extract readable article text from a news article URL."""

    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/140.0 Safari/537.36"
                )
            },
            timeout=15,
            allow_redirects=True
        )

        if response.status_code != 200:
            return f"Could not access article. HTTP status: {response.status_code}"

        final_url = response.url

        text = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=False
        )

        if not text:
            return (
                f"Could not extract readable article text "
                f"from: {final_url}"
            )

        return (
            f"Article URL: {final_url}\n\n"
            f"{text}"
        )

    except Exception as e:
        return f"Article extraction failed: {str(e)}"