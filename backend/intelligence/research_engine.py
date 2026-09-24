import json

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp.analysis.topics import extract_topics_nmf
from nlp.analysis.trends import extract_trends
from intelligence.article_analyzer import analyze_article


# Path to articles.json
BASE_DIR = Path(__file__).resolve().parents[1]
ARTICLES_FILE = BASE_DIR / "data" / "articles.json"


def load_articles():
    """
    Load articles from the static dataset.
    """

    with open(ARTICLES_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def search_articles(query, top_k=5):
    """
    Find articles most relevant to a research query
    using TF-IDF and cosine similarity.
    """

    articles = load_articles()

    documents = [article["content"] for article in articles]

    # Add the user query to the documents
    documents_with_query = [query] + documents

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(documents_with_query)

    # Compare query with every article
    similarities = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    )[0]

    # Attach similarity score to each article
    results = []

    for article, score in zip(articles, similarities):
        results.append({
            "id": article["id"],
            "title": article["title"],
            "source": article["source"],
            "date": article["date"],
            "content": article["content"],
            "similarity": float(score)
        })

    # Highest similarity first
    results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return results[:top_k]


def get_multi_article_topics(num_topics=3, num_words=5):
    """
    Extract common topics across all articles in the dataset.
    """
    articles = load_articles()
    documents = [article["content"] for article in articles]
    
    topics = extract_topics_nmf(documents, num_topics=num_topics, num_words=num_words)
    return topics


def get_related_articles(target_article_id, top_k=3):
    """
    Find related articles for a given article ID using TF-IDF similarity.
    """
    articles = load_articles()
    
    target_article = None
    for article in articles:
        if article["id"] == target_article_id:
            target_article = article
            break
            
    if not target_article:
        return []
        
    documents = [article["content"] for article in articles]
    
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    target_index = articles.index(target_article)
    
    similarities = cosine_similarity(
        tfidf_matrix[target_index:target_index+1],
        tfidf_matrix
    )[0]
    
    results = []
    for idx, (article, score) in enumerate(zip(articles, similarities)):
        if article["id"] == target_article_id:
            continue
            
        results.append({
            "id": article["id"],
            "title": article["title"],
            "similarity": float(score)
        })
        
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def analyze_research_trends():
    """
    Analyze research trends by extracting topics for each article and 
    grouping them by time period.
    """
    articles = load_articles()
    
    topics_per_article = {}
    for article in articles:
        analysis_result = analyze_article(article)
        
        keywords = [kw["word"] for kw in analysis_result.get("keywords", [])]
        
        nmf_topics = analysis_result.get("topics", [])
        for topic in nmf_topics:
            keywords.extend(topic["keywords"])
            
        topics_per_article[article["id"]] = keywords
        
    trends = extract_trends(articles, topics_per_article)
    return trends