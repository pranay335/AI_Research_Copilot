from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(text1, text2):
    """
    Calculate similarity between two texts
    using TF-IDF and cosine similarity.

    Returns a value between 0 and 1.
    """

    documents = [text1, text2]

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix[0][1]