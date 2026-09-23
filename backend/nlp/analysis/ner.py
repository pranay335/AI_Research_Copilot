import nltk
from nltk import ne_chunk, pos_tag
from nltk.tree import Tree


# Download required NLTK resource
nltk.download("maxent_ne_chunker")
nltk.download("maxent_ne_chunker_tab")
nltk.download("words")


def extract_entities(tokens):
    """
    Extract named entities from a list of tokens.
    """

    tagged_tokens = pos_tag(tokens)

    tree = ne_chunk(tagged_tokens)

    entities = []

    for subtree in tree:
        if isinstance(subtree, Tree):
            entity_name = " ".join(word for word, tag in subtree.leaves())
            entity_type = subtree.label()

            entities.append({
                "entity": entity_name,
                "type": entity_type
            })

    return entities