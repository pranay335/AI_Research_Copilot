from collections import Counter

from nlp.preprocessing.preprocess import preprocess_text
from nlp.analysis.ngrams import get_ngram_frequencies
from nlp.analysis.ner import extract_entities


def analyze_article(article):
    """
    Generate useful NLP-based intelligence for an article.
    """

    text = article["content"]

    # Preprocessing
    preprocessing_result = preprocess_text(text)

    tokens = preprocessing_result["filtered_tokens"]

    # Keyword analysis
    keyword_frequencies = Counter(tokens)

    keywords = [
        {
            "word": word,
            "frequency": frequency
        }
        for word, frequency in keyword_frequencies.most_common(10)
    ]

    # Bigram analysis
    bigram_frequencies = get_ngram_frequencies(tokens, 2)

    phrases = [
        {
            "phrase": " ".join(phrase),
            "frequency": frequency
        }
        for phrase, frequency in bigram_frequencies.most_common(10)
    ]

    # Named entities
    entities = extract_entities(tokens)

    return {
        "article_id": article["id"],
        "title": article["title"],
        "keywords": keywords,
        "phrases": phrases,
        "entities": entities
    }