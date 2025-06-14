import os
import ast
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from datasets import load_dataset


def return_dataset(dataset_name, dataset_path, remove_labels=False, tpl=1):
    """
    Parameters
    ----------
    dataset_name : str
        Name of the dataset to be loaded.
    dataset_path : str
        Path to the dataset.
    tpl : int
        Number of topics per label.
    remove_labels : bool
        Remove a percentage of labels from the dataset (set them to -1).
    input values to train test split depending on how much data is needed.
    Returns
    -------
    dict
        A dictionary with the following keys:
        train_text, train_labels, val_text, val_labels, test_text, test_labels
    """
    if dataset_name == "20ng":
        train = fetch_20newsgroups(data_home=dataset_path, subset="train")
        test = fetch_20newsgroups(data_home=dataset_path, subset="test")
        train_text, train_labels = train.data, train.target
        test_text, test_labels = test.data, test.target
        train_labels = [[l] for l in train_labels]
        test_labels = [[l] for l in test_labels]
        val_text, test_text, val_labels, test_labels = train_test_split(
            test_text, test_labels, test_size=0.33, random_state=42, shuffle=True
        )
        label_to_topic = {
            0: "alt.atheism",
            1: "comp.graphics",
            2: "comp.os.ms-windows.misc",
            3: "comp.sys.ibm.pc.hardware",
            4: "comp.sys.mac.hardware",
            5: "comp.windows.x",
            6: "misc.forsale",
            7: "rec.autos",
            8: "rec.motorcycles",
            9: "rec.sport.baseball",
            10: "rec.sport.hockey",
            11: "sci.crypt",
            12: "sci.electronics",
            13: "sci.med",
            14: "sci.space",
            15: "soc.religion.christian",
            16: "talk.politics.guns",
            17: "talk.politics.mideast",
            18: "talk.politics.misc",
            19: "talk.religion.misc",
        }

    elif dataset_name == "ag_news":
        dataset = load_dataset("ag_news", split="train", cache_dir=dataset_path)
        train_text, train_labels = dataset["text"], dataset["label"]
        dataset = load_dataset("ag_news", split="test", cache_dir=dataset_path)
        test_text, test_labels = dataset["text"], dataset["label"]
        train_text, _, train_labels, _ = train_test_split(
            train_text, train_labels, train_size=20000, random_state=42, shuffle=True
        )
        test_text, _, test_labels, _ = train_test_split(
            test_text, test_labels, train_size=5000, random_state=42, shuffle=True
        )
        train_labels = [[x] for x in train_labels]
        test_labels = [[x] for x in test_labels]
        val_text, test_text, val_labels, test_labels = train_test_split(
            test_text, test_labels, test_size=0.33, random_state=42, shuffle=True
        )
        label_to_topic = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech",
        }

    elif dataset_name == "ag_news_llm":
        dataset = load_dataset("ag_news", split="train", cache_dir=dataset_path)
        train_text, train_labels = dataset["text"], dataset["label"]
        dataset = load_dataset("ag_news", split="test", cache_dir=dataset_path)
        test_text, test_labels = dataset["text"], dataset["label"]
        train_text, _, train_labels, _ = train_test_split(
            train_text, train_labels, train_size=20000, random_state=42, shuffle=True
        )
        test_text, _, test_labels, _ = train_test_split(
            test_text, test_labels, train_size=5000, random_state=42, shuffle=True
        )
        train_labels = [[x] for x in train_labels]
        test_labels = [[x] for x in test_labels]
        val_text, test_text, val_labels, test_labels = train_test_split(
            test_text, test_labels, test_size=0.33, random_state=42, shuffle=True
        )
        label_to_topic = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech",
        }
        s_2_i = {k: v for v, k in enumerate(label_to_topic.values())}
        train_labels_text = pickle.load(
            open(os.path.join(dataset_path, "ag_news_train_preds.pkl"), "rb")
        )
        val_labels_text = pickle.load(
            open(os.path.join(dataset_path, "ag_news_val_preds.pkl"), "rb")
        )
        train_labels = [[s_2_i[x]] for x in train_labels_text]
        val_labels = [[s_2_i[x]] for x in val_labels_text]

    elif dataset_name == "dbpedia":
        dataset = load_dataset("dbpedia_14", split="train", cache_dir=dataset_path)
        train_text, train_labels = dataset["content"], np.array(dataset["label"])
        dataset = load_dataset("dbpedia_14", split="test", cache_dir=dataset_path)
        test_text, test_labels = dataset["content"], np.array(dataset["label"])
        train_text, _, train_labels, _ = train_test_split(
            train_text, train_labels, train_size=20000, random_state=42, shuffle=True
        )
        test_text, _, test_labels, _ = train_test_split(
            test_text, test_labels, train_size=5000, random_state=42, shuffle=True
        )
        train_labels = [[x] for x in train_labels]
        test_labels = [[x] for x in test_labels]
        val_text, test_text, val_labels, test_labels = train_test_split(
            test_text, test_labels, test_size=0.33, random_state=42, shuffle=True
        )
        label_to_topic = {
            0: "Company",
            1: "EducationalInstitution",
            2: "Artist",
            3: "Athlete",
            4: "OfficeHolder",
            5: "MeanOfTransportation",
            6: "Building",
            7: "NaturalPlace",
            8: "Village",
            9: "Animal",
            10: "Plant",
            11: "Album",
            12: "Film",
            13: "WrittenWork",
        }

    elif dataset_name == "self":
        (
            train_text,
            train_labels,
            val_text,
            val_labels,
            test_text,
            test_labels,
            label_to_topic,
        ) = return_self_data(dataset_path)

    elif dataset_name == "arxiv_cs":
        data = []
        with open(dataset_path + "/arxiv_cs.json", "r") as f:
            for line in f:
                data.append(json.loads(line))
        text = [d["abstract"] for d in data]
        labels = [d["categories"] for d in data]
        labels = [label.split(" ") for label in labels]
        authors = [d["authors_parsed"] for d in data]
        authors = [[" ".join(a) for a in author] for author in authors]
        authors_vocab = set([a for author in authors for a in author])
        labels_vocab = set([l for label in labels for l in label])
        # labels to one-hot vector
        authors_vocab = {a: i for i, a in enumerate(authors_vocab)}
        label_to_topic = {l: i for i, l in enumerate(labels_vocab)}
        labels = [[label_to_topic[l] for l in label] for label in labels]
        authors = [[authors_vocab[a] for a in author] for author in authors]
        label_to_topic = {i: l for l, i in label_to_topic.items()}
        authors_vocab = {i: a for a, i in authors_vocab.items()}
        # split train, val, test
        (
            train_text,
            test_text,
            train_labels,
            test_labels,
            train_authors,
            test_authors,
        ) = train_test_split(
            text, labels, authors, test_size=0.2, random_state=42, shuffle=True
        )
        (
            val_text,
            test_text,
            val_labels,
            test_labels,
            val_authors,
            test_authors,
        ) = train_test_split(
            test_text,
            test_labels,
            test_authors,
            test_size=0.5,
            random_state=42,
            shuffle=True,
        )

    else:
        raise ValueError("Dataset not supported")
    # for remoing some percentage of labels and replace it with next label int
    if remove_labels:
        print("Removing labels")
        # remove 5% of labels and replace with next label
        next_label = list(label_to_topic.keys())[-1] + 1
        next_label_one_hot = [0] * len(label_to_topic)
        next_label_one_hot[next_label] = 1
        idx = np.random.choice(len(train_labels), int(len(train_labels) * 0.05))
        train_labels[idx] = next_label_one_hot
        idx = np.random.choice(len(val_labels), int(len(val_labels) * 0.05))
        val_labels[idx] = next_label_one_hot
        idx = np.random.choice(len(test_labels), int(len(test_labels) * 0.05))
        test_labels[idx] = next_label_one_hot
        label_to_topic[next_label] = "no-label"

    # only one of the dataset has authors as of now
    if dataset_name != "arxiv_cs":
        authors_vocab = {0: "None"}
        train_authors = [[0]] * len(train_text)
        test_authors = [[0]] * len(test_text)
        val_authors = [[0]] * len(val_text)

    train_labels = [
        np.sum(np.eye(len(label_to_topic))[label], axis=0) for label in train_labels
    ]
    val_labels = [
        np.sum(np.eye(len(label_to_topic))[label], axis=0) for label in val_labels
    ]
    test_labels = [
        np.sum(np.eye(len(label_to_topic))[label], axis=0) for label in test_labels
    ]
    train_authorsn = np.zeros((len(train_authors), len(authors_vocab)))
    val_authorsn = np.zeros((len(val_authors), len(authors_vocab)))
    test_authorsn = np.zeros((len(test_authors), len(authors_vocab)))
    for i, author in enumerate(train_authors):
        for a in author:
            train_authorsn[i][a] = 1
    for i, author in enumerate(val_authors):
        for a in author:
            val_authorsn[i][a] = 1
    for i, author in enumerate(test_authors):
        for a in author:
            test_authorsn[i][a] = 1

    # everything to numpy
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)
    train_authors = train_authorsn
    val_authors = val_authorsn
    test_authors = test_authorsn
    new_train_labels = np.zeros((train_labels.shape[0], tpl * train_labels.shape[1]))
    new_val_labels = np.zeros((val_labels.shape[0], tpl * val_labels.shape[1]))
    new_test_labels = np.zeros((test_labels.shape[0], tpl * test_labels.shape[1]))

    for i in range(train_labels.shape[0]):
        idx = np.where(train_labels[i] == 1)[0]
        for j in idx:
            new_train_labels[i][j * tpl : (j + 1) * tpl] = 1
    for i in range(val_labels.shape[0]):
        idx = np.where(val_labels[i] == 1)[0]
        for j in idx:
            new_val_labels[i][j * tpl : (j + 1) * tpl] = 1
    for i in range(test_labels.shape[0]):
        idx = np.where(test_labels[i] == 1)[0]
        for j in idx:
            new_test_labels[i][j * tpl : (j + 1) * tpl] = 1
    train_labels = new_train_labels
    val_labels = new_val_labels
    test_labels = new_test_labels
    # duplicate labels_to_topic to tpl (topics per label)
    topics_per_label = {}
    for label in label_to_topic:
        for i in range(label * tpl, (label + 1) * tpl):
            topics_per_label[i] = label_to_topic[label]
    label_names = list(topics_per_label.values())

    return {
        "train_text": train_text,
        "train_labels": train_labels,
        "train_authors": train_authors,
        "val_text": val_text,
        "val_labels": val_labels,
        "val_authors": val_authors,
        "test_text": test_text,
        "test_labels": test_labels,
        "test_authors": test_authors,
        "label_to_topic": label_to_topic,
        "label_names": label_names,
        "authors_vocab": authors_vocab,
    }


def return_self_data(path):
    """This is a utility function to process an external dataset.

    Just save your dataset in a folder with train.csv, val.csv, and test.csv.
    Each of these files should contain following IDs:
    "text": The text document.
    "label": document labels (if they exist, in text form).
    Alert: Make sure all your splits contain all the labels available.

    Args:
        path (path): Path of the directory which contains train.csv, val.csv, and test.csv
    """
    train = pd.read_csv(os.path.join(path, "train.csv"))
    val = pd.read_csv(os.path.join(path, "val.csv"))
    test = pd.read_csv(os.path.join(path, "test.csv"))
    train_text = train["text"].to_list()
    val_text = val["text"].to_list()
    test_text = test["text"].to_list()
    try:
        train_labels = []
        for label in train["label"].to_list():
            train_labels.append(ast.literal_eval(label))
        val_labels = []
        for label in val["label"].to_list():
            val_labels.append(ast.literal_eval(label))
        test_labels = []
        for label in test["label"].to_list():
            test_labels.append(ast.literal_eval(label))
        train_label_vocab = list(set([word for labels in train_labels for word in labels]))
        val_label_vocab = train_label_vocab
        test_label_vocab = train_label_vocab
        train_label_vocab.sort()
        val_label_vocab.sort()
        test_label_vocab.sort()
        assert train_label_vocab == val_label_vocab == test_label_vocab
        label_to_topic = {
            i: train_label_vocab[i] for i in range(len(train_label_vocab))
        }
        l2i = {v: i for i, v in label_to_topic.items()}
        train_labels = [[l2i[word] for word in label] for label in train_labels]
        val_labels = [[l2i[word] for word in label] for label in val_labels]
        test_labels = [[l2i[word] for word in label] for label in test_labels]
    except:
        train_labels = [[0]] * len(train_text)
        val_labels = [[0]] * len(val_text)
        test_labels = [[0]] * len(test_text)
        label_to_topic = {0: "none"}
    return (
        train_text,
        train_labels,
        val_text,
        val_labels,
        test_text,
        test_labels,
        label_to_topic,
    )
