"""\nData loading and preprocessing utilities."""

import os
import torch
import pickle
import numpy as np
from ._datasets import return_dataset
from ._embedding import return_word_embeddings
from tokenease import Pipe
from ._data_classes import DataClass
from sentence_transformers import SentenceTransformer

__all__ = [
    "return_prepared_data",
    "return_dataloaders",
]


def return_prepared_data(dataset_name, data_path, remove_labels=False, tpl=1, **kwargs):
    """Return the prepared dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load.
    data_path : str
        Path to the data directory.

    Returns
    -------
    dict
        Dictionary containing the bag of words dataset.
        keys: train_bow, train_labels, val_bow, val_labels, test_bow, test_labels, vocab, vectorizer
    """
    prepared_data_path = os.path.join(data_path, dataset_name + "_prepared_data.pkl")
    # try to load the prepared data
    try:
        prepared_data = pickle.load(open(prepared_data_path, "rb"))
        print("Prepared data loaded.")
        return prepared_data
    except FileNotFoundError:
        pass
    # return dataset
    dataset = return_dataset(dataset_name, data_path, remove_labels, tpl)
    # tokenize the text (save the word level text separately)
    tokenizer = Pipe(
        remove_stop_words=True, min_df=kwargs["min_df"], max_df=kwargs["max_df"]
    )
    tokenizer.save(os.path.join(kwargs["exp_path"], "tokenizer.joblib"))
    train_bow, train_text = tokenizer.fit_transform(dataset["train_text"])
    val_bow, val_text = tokenizer.transform(dataset["val_text"])
    test_bow, test_text = tokenizer.transform(dataset["test_text"])
    # normalize the bag of words
    # train_bow = train_bow / np.sum(train_bow, axis=1)[:, None]
    # val_bow = val_bow / np.sum(val_bow, axis=1)[:, None]
    # test_bow = test_bow / np.sum(test_bow, axis=1)[:, None]
    train_wl_text = [x.split(tokenizer.seperator) for x in train_text]
    val_wl_text = [x.split(tokenizer.seperator) for x in val_text]
    test_wl_text = [x.split(tokenizer.seperator) for x in test_text]
    vocabulary = tokenizer.vocabulary
    # return word embeddings
    vocab_embeddings = return_word_embeddings(vocabulary)
    vocab_embeddings = torch.from_numpy(vocab_embeddings).float()
    prepared_data = {
        "train_bow": train_bow.astype(np.float32),
        "train_text": dataset["train_text"],
        "train_labels": dataset["train_labels"],
        "train_authors": dataset["train_authors"],
        "val_bow": val_bow.astype(np.float32),
        "val_text": dataset["val_text"],
        "val_labels": dataset["val_labels"],
        "val_authors": dataset["val_authors"],
        "test_bow": test_bow.astype(np.float32),
        "test_text": dataset["test_text"],
        "test_labels": dataset["test_labels"],
        "test_authors": dataset["test_authors"],
        "vocab": vocabulary,
        "vocab_embeddings": vocab_embeddings,
        "label_to_topic": dataset["label_to_topic"],
        "label_names": dataset["label_names"],
        "authors_vocab": dataset["authors_vocab"],
        "text": train_wl_text,
        "train_wl_text": train_wl_text,
        "val_wl_text": val_wl_text,
        "test_wl_text": test_wl_text,
    }
    # save the prepared data
    pickle.dump(prepared_data, open(prepared_data_path, "wb"))
    return prepared_data


def return_dataloaders(model_name, prepared_data, batch_size=32, **kwargs):
    """
    Args:
        model_name: str
        prepared_data: dict
        batch_size: int
    Returns:
        train_dl, val_dl, test_dl: DataLoader objects
    """
    train_kwargs = {}
    val_kwargs = {}
    test_kwargs = {}
    train_kwargs["placeholder"] = np.zeros(len(prepared_data["train_bow"]))
    val_kwargs["placeholder"] = np.zeros(len(prepared_data["val_bow"]))
    test_kwargs["placeholder"] = np.zeros(len(prepared_data["test_bow"]))
    if "vae" or "scholar" in model_name:
        train_kwargs["bow"] = prepared_data["train_bow"]
        val_kwargs["bow"] = prepared_data["val_bow"]
        test_kwargs["bow"] = prepared_data["test_bow"]
    if "labels" or "scholar" in model_name:
        train_kwargs["labels"] = prepared_data["train_labels"]
        val_kwargs["labels"] = prepared_data["val_labels"]
        test_kwargs["labels"] = prepared_data["test_labels"]
    if "authors" in model_name:
        train_kwargs["authors"] = prepared_data["train_authors"]
        val_kwargs["authors"] = prepared_data["val_authors"]
        test_kwargs["authors"] = prepared_data["test_authors"]
    if ("context" in model_name) or ("llm" in model_name):
        model = SentenceTransformer(kwargs["sentence_transformer_name"])
        train_doc_emb = model.encode(
            prepared_data["train_text"], device=kwargs["device"]
        )
        val_doc_emb = model.encode(prepared_data["val_text"], device=kwargs["device"])
        test_doc_emb = model.encode(prepared_data["test_text"], device=kwargs["device"])
        train_kwargs["doc_emb"] = train_doc_emb
        val_kwargs["doc_emb"] = val_doc_emb
        test_kwargs["doc_emb"] = test_doc_emb
    train_ds = DataClass(**train_kwargs)
    val_ds = DataClass(**val_kwargs)
    test_ds = DataClass(**test_kwargs)

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_dl, val_dl, test_dl
