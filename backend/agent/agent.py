from pathlib import Path
from dotenv import load_dotenv

# Load .env from the backend folder
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq

from tools.news import search_news


# Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# Available tools
tools = [search_news]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)


# Agent node
def agent(state: MessagesState):
    messages = state["messages"]

    system_message = (
        "You are an AI Research Copilot. "
        "You help users find and understand current technology and AI information. "
        "You can use the available search_news tool to find recent news. "
        "Use only the tools provided to you. "
        "Do not attempt to use tools that are not available. "
        "When the user asks for current news, use search_news first. "
        "After receiving the tool results, provide a clear and concise answer."
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


# Create LangGraph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))

# Starting point
builder.add_edge(START, "agent")

# Decide whether the agent needs a tool
builder.add_conditional_edges(
    "agent",
    tools_condition
)

# After tool execution, return to the agent
builder.add_edge("tools", "agent")


# Compile graph
graph = builder.compile()