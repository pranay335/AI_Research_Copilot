import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# Initialize NLP tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))


def preprocess_text(text: str):
    """
    Perform basic NLP preprocessing.

    Steps:
    1. Convert text to lowercase
    2. Tokenize
    3. Filtration
    4. Remove stopwords
    5. Stemming
    6. Lemmatization
    """

    # Lowercase
    text = text.lower()

    # Remove unwanted characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Filtration
    filtered_tokens = [
        token
        for token in tokens
        if token.isalpha()
    ]

    # Stopword removal
    filtered_tokens = [
        token
        for token in filtered_tokens
        if token not in stop_words
    ]

    # Stemming
    stemmed_tokens = [
        stemmer.stem(token)
        for token in filtered_tokens
    ]

    # Lemmatization
    lemmatized_tokens = [
        lemmatizer.lemmatize(token)
        for token in filtered_tokens
    ]

    return {
        "original": text,
        "tokens": tokens,
        "filtered_tokens": filtered_tokens,
        "stemmed_tokens": stemmed_tokens,
        "lemmatized_tokens": lemmatized_tokens
    }