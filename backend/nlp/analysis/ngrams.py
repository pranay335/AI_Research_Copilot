from collections import Counter


def generate_ngrams(tokens, n):
    """
    Generate n-grams from a list of tokens.

    Example:
    tokens = ["ai", "agents", "are", "useful"]

    n = 2
    Output:
    [("ai", "agents"), ("agents", "are"), ("are", "useful")]
    """

    if n <= 0:
        return []

    return [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]


def get_ngram_frequencies(tokens, n):
    """
    Generate n-grams and calculate their frequencies.
    """

    ngrams = generate_ngrams(tokens, n)

    return Counter(ngrams)


def analyze_ngrams(tokens):
    """
    Generate unigram, bigram and trigram frequencies.
    """

    return {
        "unigrams": get_ngram_frequencies(tokens, 1),
        "bigrams": get_ngram_frequencies(tokens, 2),
        "trigrams": get_ngram_frequencies(tokens, 3)
    }