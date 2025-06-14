import torch
from torch.distributions import Dirichlet
from ._base_model import BaseModel
from .vae_helper._decoder import Decoder, SharedETMDecoder


class SCHOLAR(BaseModel):
    def __init__(self, encoder, sampler, decoder, nll_loss, kld_loss, **kwargs) -> None:
        super().__init__()
        self.encoder = encoder
        self.sampler = sampler
        self.decoder = decoder
        self.alpha = kwargs["alpha"]
        self.num_topics = kwargs["num_topics"]
        self.nll_loss = nll_loss
        self.kld_loss = kld_loss
        self.model_name = kwargs["model_name"]
        self.kwargs = kwargs
        # check if authors are present (if so, add author decoder)
        if "authors" in self.model_name:
            self.i2a = kwargs["i2a"]  # index to author
            self.num_authors = len(self.i2a)  # number of authors
            self.author_decoder_name = self.model_name.split("-")[-1]
            # author decoder
            if self.author_decoder_name == "lin":
                self.author_decoder = Decoder(self.num_topics, self.num_authors)
            elif self.author_decoder_name == "etm":
                self.author_decoder = SharedETMDecoder(
                    out_features=self.num_authors,
                )
            else:
                raise ValueError(
                    f"Author decoder {self.author_decoder_name} not implemented."
                )

        if "labels" in self.model_name:
            self.eps = kwargs["eps"]

        self.classifier = torch.nn.Linear(
            self.num_topics, len(self.kwargs["label_names"])
        )

    def forward(
        self,
        x: torch.Tensor,
        **kwargs,
    ):
        # Encode the input
        labels = kwargs["labels"]
        x_in = torch.cat([x, labels], dim=1)
        x_in = x_in.float()
        dist_params = self.encoder(x_in)
        prior_alpha = torch.ones((x.shape[0], self.num_topics), device=x.device)
        if "labels" in self.model_name:
            prior_alpha *= self.eps
            act = torch.ones((x.shape[0], self.num_topics), device=x.device) * self.eps
            idx = labels == 1
            act[idx] = 1
            prior_alpha[idx] = self.alpha
            dist_params *= act
        # Sample the latent variable
        z = self.sampler(dist_params)
        # Decode the latent variable
        x_hat = self.decoder(z)
        # compute loss
        kl_d = self.kld_loss(Dirichlet(dist_params), Dirichlet(prior_alpha))
        nll_loss = self.nll_loss(x, x_hat)
        if "authors" in self.model_name:
            # Decoder the authors
            a = kwargs["authors"]
            if self.a_decoder == "lin":
                a_hat = self.author_decoder(z)
            elif self.a_decoder == "etm":
                a_hat = self.author_decoder(z, te=self.decoder.topic_embeddings)
            # compute author loss
            nll_loss += self.nll_loss(a, a_hat)
        nll_loss += self.nll_loss(labels, self.classifier(z))
        return x, dist_params, z, x_hat, kl_d, nll_loss

    def files_to_save(self):
        file_dict = {}
        json_dict = {}
        topics = self.decoder.get_topics()
        file_dict["topics.pkl"] = topics
        topic_word_list = self.get_topic_words(topics, self.kwargs["i2w"])
        json_dict["topic_words"] = topic_word_list
        # label list
        json_dict["label_names"] = self.kwargs["label_names"]
        # topic to label
        topic_to_label_dist = self.classifier.weight.data.cpu().T
        # take softmax
        topic_to_label = torch.nn.functional.softmax(topic_to_label_dist, dim=1).numpy()
        file_dict["topic_to_label.pkl"] = topic_to_label
        # (label_name, prob) per topic
        # topic_label_list = []
        # for i, topic in enumerate(topic_to_label):
        #     topic_label_list.append(
        #         [
        #             (self.kwargs["label_names"][j], str(prob))
        #             for j, prob in enumerate(topic)
        #         ]
        #     )
        # json_dict["topic_labels_dist"] = topic_label_list
        # topic labels (argmax)
        topic_labels = torch.argmax(topic_to_label_dist, dim=1).numpy()
        # get label names
        topic_labels = [self.kwargs["label_names"][i] for i in topic_labels]
        json_dict["topic_labels_scholar"] = topic_labels

        if "authors" in self.model_name:
            # top authors per topic
            if self.author_decoder_name == "lin":
                author_topics = self.author_decoder.get_topics()
            elif self.author_decoder_name == "etm":
                author_topics = self.author_decoder.get_topics(
                    te=self.decoder.topic_embeddings
                )
                # author_embeddings
                author_embeddings = (
                    self.author_decoder.author_embeddings.weight.data.cpu().numpy().T
                )
                file_dict["author_embeddings.pkl"] = author_embeddings
            file_dict["author_topics.pkl"] = author_topics
            author_topic_list = self.get_topic_words(author_topics, self.i2a)
            json_dict["author_topic_words"] = author_topic_list

        # topic embeddings
        if "etm" in self.model_name:
            topic_embeddings = self.decoder.topic_embeddings.weight.data.cpu().numpy().T
            file_dict["topic_embeddings.pkl"] = topic_embeddings
            word_embeddings = self.decoder.word_embeddings.weight.data.cpu().numpy().T
            file_dict["word_embeddings.pkl"] = word_embeddings
        file_dict["topics.json"] = json_dict
        return file_dict
