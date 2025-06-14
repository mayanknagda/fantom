import torch
import torch.nn as nn
import numpy as np


class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def get_topic_words(self, topics, i2w, n=10):
        topic_list = []
        for i, topic in enumerate(topics):
            top_k = np.argsort(topic)[::-1][:n]
            topic_list.append([i2w[idx] for idx in top_k])
        return topic_list

    def files_to_save(self):
        raise NotImplementedError
