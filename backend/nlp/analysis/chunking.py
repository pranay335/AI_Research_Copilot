import nltk

from nltk import RegexpParser


def get_chunks(pos_tags):
    """
    Extract noun phrases and other meaningful chunks
    using POS tags.
    """

    grammar = """
        NP: {<DT>?<JJ.*>*<NN.*>+}
    """

    chunk_parser = RegexpParser(grammar)

    return chunk_parser.parse(pos_tags)