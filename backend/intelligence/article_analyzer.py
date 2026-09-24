from nlp.preprocessing.preprocess import preprocess_text
from nlp.analysis.ngrams import get_ngram_frequencies
from nlp.analysis.pos_tagging import get_pos_tags
from nlp.analysis.ner import extract_entities
from nlp.analysis.similarity import calculate_similarity
from nlp.summarization.summarizer import summarize_article
from nlp.analysis.topics import extract_topics_nmf


def analyze_article(article):
    """
    Run the complete NLP analysis pipeline for one article.
    """

    title = article["title"]
    content = article["content"]

    # -------------------------
    # 1. Preprocessing
    # -------------------------
    preprocessing = preprocess_text(content)

    tokens = preprocessing["filtered_tokens"]

    # -------------------------
    # 2. N-Grams
    # -------------------------
    bigram_frequencies = get_ngram_frequencies(tokens, 2)

    important_phrases = [
        {
            "phrase": " ".join(phrase),
            "frequency": frequency
        }
        for phrase, frequency in bigram_frequencies.most_common(10)
    ]

    # -------------------------
    # 3. POS Tagging
    # -------------------------
    pos_tags = get_pos_tags(tokens)

    # -------------------------
    # 4. Named Entity Recognition
    # -------------------------
    entities = extract_entities(tokens)

    # -------------------------
    # 5. Keywords
    # -------------------------
    from collections import Counter

    keyword_frequencies = Counter(tokens)

    keywords = [
        {
            "word": word,
            "frequency": frequency
        }
        for word, frequency in keyword_frequencies.most_common(10)
    ]

    # -------------------------
    # 6. Topic Extraction
    # -------------------------
    # Treat sentences as documents for single-article topic modeling
    sentences = content.split('.')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    # Extract topics using NMF on the article's sentences
    topics = extract_topics_nmf(sentences, num_topics=2, num_words=3)

    # -------------------------
    # 7. Summarization
    # -------------------------
    summary = summarize_article(
        title,
        content
    )

    return {
        "article_id": article["id"],
        "title": title,
        "keywords": keywords,
        "topics": topics,
        "phrases": important_phrases,
        "entities": entities,
        "pos_tags": pos_tags,
        "summary": summary
    }


def compare_articles(article1, article2):
    """
    Calculate similarity between two articles.
    """

    return calculate_similarity(
        article1["content"],
        article2["content"]
    )