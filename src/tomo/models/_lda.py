import os
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary


class LDA:
    def __init__(self, train_texts, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        i2w = kwargs["i2w"]
        w2i = {v: k for k, v in i2w.items()}
        new_text = []
        for doc in train_texts:
            new_doc = []
            for word in doc:
                if word in w2i:
                    new_doc.append(word)
            new_text.append(new_doc)
        self.dictionary = Dictionary(new_text)
        self.corpus = [self.dictionary.doc2bow(text) for text in new_text]

    def run(self):
        self.model = LdaModel(
            self.corpus,
            num_topics=self.kwargs["num_topics"],
            id2word=self.dictionary,
            random_state=self.kwargs["random_state"],
        )

    def files_to_save(self):
        self.model.save(os.path.join(self.kwargs["exp_path"], "lda.model"))
        file_dict = {}
        json_dict = {}
        topics = self.model.show_topics(
            num_topics=self.kwargs["num_topics"], formatted=False
        )
        topics_words = [([wd[0] for wd in tp[1]]) for tp in topics]
        json_dict["topic_words"] = topics_words
        file_dict["topics.json"] = json_dict
        return file_dict
