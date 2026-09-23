import json

from nlp.preprocessing.preprocess import preprocess_text
from nlp.analysis.pos_tagging import get_pos_tags


# Load static articles
with open("data/articles.json", "r", encoding="utf-8") as file:
    articles = json.load(file)


# Use the first article
article = articles[0]

# Get article content
text = article["content"]


# Run preprocessing
result = preprocess_text(text)

# Get filtered tokens
tokens = result["filtered_tokens"]


# POS tagging
pos_tags = get_pos_tags(tokens)


print("\n===== POS TAGGING =====")

for word, tag in pos_tags:
    print(f"{word} -> {tag}")