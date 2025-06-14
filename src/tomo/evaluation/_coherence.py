from gensim.models import CoherenceModel
from gensim.corpora import Dictionary


def return_coherence(topics: list, text: list, metric: str):
    """
    Return coherence score for a given list of topics.
    """
    dictionary = Dictionary(text)
    cm = CoherenceModel(
        topics=topics,
        texts=text,
        dictionary=dictionary,
        coherence=metric,
        processes=-1,
        topn=10,
    )
    return cm.get_coherence_per_topic(), cm.get_coherence()
