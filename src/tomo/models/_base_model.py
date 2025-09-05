"""\nModel definitions and helpers for topic models.\n\nThis module is part of the `tomo` topic modeling library.\n"""

import torch
import torch.nn as nn
import numpy as np


class BaseModel(nn.Module):
    """\n    Class `BaseModel`.\n\n    Args:\n    nn.Module: Description.\n\n    Returns: Description.\n"""

    def __init__(self) -> None:
        super().__init__()
        pass

    def get_topic_words(self, topics, i2w, n=10):
        """\n        Function `get_topic_words`.\n    \n        Args:\n        topics: Description.\n        i2w: Description.\n        n: Description.\n    \n        Returns: Description.\n"""
        topic_list = []
        for i, topic in enumerate(topics):
            top_k = np.argsort(topic)[::-1][:n]
            topic_list.append([i2w[idx] for idx in top_k])
        return topic_list

    def files_to_save(self):
        """\n        Function `files_to_save`.\n    \n        Returns: Description.\n"""
        raise NotImplementedError
