from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


def extract_topics_nmf(documents, num_topics=3, num_words=5):
    """
    Extract topics from a list of documents using NMF and TF-IDF.
    """
    if not documents:
        return []

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

    n_components = min(num_topics, len(documents), tfidf_matrix.shape[1])
    
    if n_components == 0:
        return []

    nmf_model = NMF(n_components=n_components, random_state=42)
    nmf_model.fit(tfidf_matrix)

    feature_names = vectorizer.get_feature_names_out()
    
    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_features_ind = topic.argsort()[:-num_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topics.append({
            "topic_id": topic_idx + 1,
            "keywords": top_features
        })
        
    return topics
