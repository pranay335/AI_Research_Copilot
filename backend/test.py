from agent.agent import graph
from intelligence.research_engine import get_multi_article_topics, get_related_articles, analyze_research_trends
from intelligence.article_analyzer import analyze_article
from intelligence.research_engine import load_articles


response = graph.invoke(
    {
        "messages": [
            (
                "user",
                "What information do we have about AI agents?"
            )
        ]
    }
)


print("\n===== RESEARCH COPILOT =====")

for message in response["messages"]:

    if message.content:
        # Handle unicode encoding for windows console
        clean_content = message.content.encode('ascii', 'replace').decode('ascii')
        print("\n" + clean_content)


print("\n===== TESTING NEW FEATURES =====")

print("\n1. Multi-Article Topics:")
topics = get_multi_article_topics(num_topics=3, num_words=5)
for topic in topics:
    print(f"Topic {topic['topic_id']}: {', '.join(topic['keywords'])}")

print("\n2. Related Article Detection (for article ID 1):")
related = get_related_articles(target_article_id=1, top_k=2)
for article in related:
    print(f"ID {article['id']}: {article['title']} (Similarity: {article['similarity']:.2f})")

print("\n3. Single Article Analysis (topics included):")
articles = load_articles()
if articles:
    analysis = analyze_article(articles[0])
    print(f"Analyzed Article: {analysis['title']}")
    print("Extracted Topics (from single article):")
    for topic in analysis['topics']:
        print(f" - {topic}")

print("\n4. Research Trends:")
trends = analyze_research_trends()
import json
print(json.dumps(trends, indent=2))


print("\n===== TESTING STAGE 5.1: MEMORY =====")
from memory.memory import initialize_database, save_research, get_research_history, get_research_by_query

print("\n1. Initializing Database...")
initialize_database()
print("Database initialized successfully.")

print("\n2. Saving sample research record...")
sample_query = "What information do we have about AI agents?"
sample_topics = ["AI agents", "software development", "multimodal AI"]
sample_articles = [{"id": 1, "title": "AI Agents Transform Enterprise Software"}]
save_research(sample_query, sample_topics, sample_articles)
print("Saved successfully.")

print("\n3. Retrieving recent research history...")
history = get_research_history(limit=5)
print(json.dumps(history, indent=2))

print("\n4. Searching saved query ('AI agents')...")
search_results = get_research_by_query("AI agents", limit=5)
print(json.dumps(search_results, indent=2))


print("\n===== TESTING STAGE 5.2: AGENT MEMORY =====")

def run_agent(query):
    response = graph.invoke({"messages": [("user", query)]})
    for message in response["messages"]:
        if message.content:
            clean_content = message.content.encode('ascii', 'replace').decode('ascii')
            print("\n" + clean_content)
    print("\n" + "-"*40)

print("\n1. Search existing research memory:")
run_agent("Do we have any previous research about AI agents?")

print("\n2. Save a new research record through the memory tool:")
run_agent("Please save a research record for 'multimodal systems' with topic 'multimodal' and article ID 4 'Multimodal AI Connects Text, Images, and Audio'.")

print("\n3. Search for that record again:")
run_agent("Do we have any previous research saved about multimodal systems?")

print("\n4. Confirm that the existing research search still works:")
run_agent("Can you search the article collection for 'speech technology'?")

