def return_topic_diversity(topics: list):
    """
    Return topic diversity score for a given list of topics.
    """
    unique_words = set()
    for topic in topics:
        for word in topic:
            if word not in unique_words:
                unique_words.add(word)
    return len(unique_words) / (len(topics) * len(topics[0]))
