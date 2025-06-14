from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer


class BTopic:
    def __init__(self, train_texts, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        i2w = kwargs["i2w"]
        w2i = {v: k for k, v in i2w.items()}
        vectorizer_model = CountVectorizer(stop_words="english")
        sentence_model = SentenceTransformer(self.kwargs["sentence_transformer_name"])
        self.model = BERTopic(
            embedding_model=sentence_model,
            vectorizer_model=vectorizer_model,
            nr_topics=kwargs["num_topics"],
        )
        self.train_texts = train_texts

    def run(self):
        self.model.fit_transform(self.train_texts)

    def files_to_save(self):
        topics = self.model.get_topics()
        topics_words = []
        for _, values in topics.items():
            topics_words.append([tup[0] for tup in values])
        file_dict = {}
        json_dict = {}
        json_dict["topic_words"] = topics_words
        file_dict["topics.json"] = json_dict
        return file_dict
