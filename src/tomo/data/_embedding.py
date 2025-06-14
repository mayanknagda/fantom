import numpy as np
import gensim.downloader as api


def return_word_embeddings(vocab):
    glove_vectors = api.load("glove-wiki-gigaword-300")
    word_embeddings = np.zeros((len(vocab), 300))
    for i, word in enumerate(vocab):
        if word in glove_vectors:
            word_embeddings[i] = glove_vectors[word]
    return word_embeddings
