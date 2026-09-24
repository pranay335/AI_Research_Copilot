from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


def summarize_article(title, content):
    prompt = f"""
You are an AI Research Copilot.

Summarize the following article clearly and accurately.

Title:
{title}

Article:
{content}

Requirements:
- Keep the summary concise.
- Include the main idea and important points.
- Do not add information that is not present in the article.
- Use simple and clear language.
"""

    response = llm.invoke(prompt)

    return response.content