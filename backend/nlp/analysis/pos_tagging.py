import nltk
from nltk import pos_tag

# Download POS tagging resources
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")


def get_pos_tags(tokens):
    """
    Perform Part-of-Speech tagging on a list of tokens.
    """

    return pos_tag(tokens)