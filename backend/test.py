import json

from nlp.analysis.similarity import calculate_similarity


# Load static articles
with open("data/articles.json", "r", encoding="utf-8") as file:
    articles = json.load(file)


# Select two articles
article1 = articles[0]
article2 = articles[1]


# Get article content
text1 = article1["content"]
text2 = article2["content"]


# Calculate similarity
score = calculate_similarity(text1, text2)


print("\n===== TEXT SIMILARITY =====")

print(f"Article 1: {article1['title']}")
print(f"Article 2: {article2['title']}")

print(f"\nSimilarity Score: {score:.4f}")