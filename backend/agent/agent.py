from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from intelligence.research_engine import search_articles


# ==========================================
# Load environment variables
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


# ==========================================
# LLM
# ==========================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# ==========================================
# Research Tool
# ==========================================

@tool
def search_research(query: str) -> str:
    """
    Search the internal research article collection
    for articles relevant to the user's query.
    """

    results = search_articles(query, top_k=5)

    if not results:
        return "No relevant articles found."

    output = []

    for article in results:
        output.append(
            f"Title: {article['title']}\n"
            f"Source: {article['source']}\n"
            f"Date: {article['date']}\n"
            f"Similarity: {article['similarity']:.4f}\n"
            f"Content: {article['content']}"
        )

    return "\n\n---\n\n".join(output)


from memory.memory import get_research_by_query, save_research

@tool
def search_research_memory(query: str) -> str:
    """
    Search previous research history.
    Use this when the user asks about previous research or history.
    """
    results = get_research_by_query(query, limit=5)
    
    if not results:
        return "No previous research found."
        
    output = []
    for row in results:
        output.append(
            f"Query: {row['query']}\n"
            f"Topics: {row['topics']}\n"
            f"Articles: {row['articles']}\n"
            f"Date: {row['created_at']}"
        )
        
    return "\n\n---\n\n".join(output)

@tool
def save_research_memory(query: str, topics: list, articles: list) -> str:
    """
    Save a completed research interaction to memory.
    topics should be a list of strings.
    articles should be a list of compact references (e.g. dicts with 'id' and 'title').
    """
    save_research(query, topics, articles)
    return "Research saved to memory successfully."


# ==========================================
# Tools
# ==========================================

tools = [search_research, search_research_memory, save_research_memory]

llm_with_tools = llm.bind_tools(tools)


# ==========================================
# Agent
# ==========================================

def agent(state: MessagesState):
    """
    Main LangGraph agent node.
    """

    messages = state["messages"]

    system_message = (
        "You are an AI Research Copilot. "
        "You help users find and understand research and technology information. "
        "Use search_research for information from the research article collection. "
        "Use search_research_memory when the user asks about previous research/interactions. "
        "Save a research interaction after a meaningful research request has been completed using save_research_memory. "
        "Do not save casual conversation. "
        "After receiving the retrieved articles, synthesize their information "
        "into a clear and concise research answer. "
        "Do not simply repeat the raw article content. "
        "Use only information present in the retrieved articles. "
        "Do not invent information."
    )

    response = llm_with_tools.invoke(
        [
            ("system", system_message),
            *messages
        ]
    )

    return {
        "messages": [response]
    }


# ==========================================
# LangGraph
# ==========================================

builder = StateGraph(MessagesState)

builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    tools_condition
)

builder.add_edge("tools", "agent")

graph = builder.compile()