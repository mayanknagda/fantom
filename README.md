# tomo — Topic Modeling Toolkit

A lightweight, modular library for classic and neural topic models. It supports:
- **LDA** (via `gensim`)
- **BERTopic** (via `bertopic` + sentence-transformers)
- **VAE-based topic models** (configurable encoders/samplers/decoders also with labels/authors (FANToM) )
- **SCHOLAR** (supervised topic model with labels)

It includes utilities for **dataset handling**, **preprocessing**, **training loops**, and **evaluation** (coherence, diversity), plus a ready-to-use experiment script.

---

## Installation

### 1) Create an environment
```bash
# Python 3.10 recommended
conda create -n tomo python=3.10 -y
conda activate tomo
```

### 2) Install dependencies (Poetry)
```bash
pip install poetry
poetry install
```
Or using pip directly:
```bash
pip install -r <(poetry export -f requirements.txt --without-hashes)
# or install the listed deps manually from pyproject.toml
```

### 3) spaCy model for English
```bash
python -m spacy download en_core_web_sm
```

---

## Quickstart (Python API)

```python
from tomo.run import run

config = dict(
    # Logging
    wandb=False,
    project_name="topic-models",
    wandb_path="./runs",

    # Data
    dataset_name="20ng",            # or "ag_news", "dbpedia", or a custom name
    dataset_path="./datasets/20ng", # for custom datasets
    remove_labels=False,
    tpl=1,                          # tokens-per-line used in tokenease

    # Output
    exp_path="./runs/exp1",
    device="cuda:0",                # or "cpu"

    # Model
    model_name="lda",               # "lda", "bertopic", "vae-...", "scholar-..."
    num_topics=20,                  # set null/None to infer from labels where applicable
    doc_emb_model="all-MiniLM-L6-v2",
    doc_emb_dim=384,
    eps=1e-8, alpha=0.01, beta=2.0, # priors/hyperparams for VAE/SCHOLAR

    # Training
    batch_size=128,
    lr=1e-3,
    epochs=50,
    random_state=0,
)

model, summary = run(**config)
print(summary)
```
---

## Quickstart (from YAML)

Use the helper script to run from a config file:
```bash
python experiments/run_from_template.py path/to/config.yaml
```
See [`config/template.yaml`](config/template.yaml) for available fields. Typical keys:

```yaml
# --- logging ---
wandb: false
project_name: "topic-models"
wandb_path: "./runs"

# --- data ---
dataset_name: "20ng"     # or ag_news/dbpedia or your custom dataset name
dataset_path: "./datasets/my_corpus"  # used if dataset_name is custom
remove_labels: false
tpl: 1                   # tokenease tokens-per-line

# --- output ---
exp_path: "./runs/exp1"
device: "cuda:0"

# --- model ---
model_name: "lda"        # "lda", "bertopic", "vae-...", "scholar-..."
num_topics: 20           # null/None to infer from labels for SCHOLAR
doc_emb_model: "all-MiniLM-L6-v2"
doc_emb_dim: 384
eps: 1e-8
beta: 2.0
alpha: 0.01

# --- training ---
batch_size: 128
lr: 0.001
epochs: 100
random_state: 0
```

---

## Supported Models in `tomo`

### 1. Classical Models
- **LDA**
  - `model_name: "lda"`
  - Implementation: `gensim.models.LdaModel`
  - No encoder/sampler/decoder — classic probabilistic topic model.

- **BERTopic**
  - `model_name: "bertopic"`
  - Implementation: `BERTopic` + `SentenceTransformer`
  - Uses document embeddings + clustering (no encoder/sampler/decoder).

### 2. Variational Autoencoder (VAE) Family
General naming convention:
```(vae)-(labels-authors-ecrtm)-(encoder-sampler-decoder)
```
- **Encoders**
  - `lin` → linear BoW encoder
  - `context` → concatenated BoW + SentenceTransformer embeddings
  - `llm` → LLM embeddings only

- **Samplers**
  - `dir_pathwise` → Dirichlet pathwise gradient sampler
  - `dir_rsvi` → Dirichlet Rejection Sampling Variational Inference (RSVI)

- **Decoders**
  - `lin` → linear decoder
  - `etm` → Embedding Topic Model (ETM) decoder

- **Special Flags**
  - `labels` → supervised with document labels
  - `authors` → author-topic model (uses author metadata)
  - `ecrtm` → adds ECR regularization (topic/word embedding alignment)

**Examples:**
- `vae-lin-dir_pathwise-etm`
- `vae-context-dir_rsvi-lin`
- `vae-llm-dir_rsvi-etm`
- `vae-authors-lin-dir_pathwise-lin`
- `vae-ecrtm-lin-dir_rsvi-etm`

### 3. SCHOLAR (Supervised VAE)
General naming convention:
```scholar-(labels-authors)-(encoder-sampler-decoder)```
- **Encoder**
  - Always `lin` (linear + label embeddings)

- **Samplers**
  - `dir_pathwise`
  - `dir_rsvi`

- **Decoders**
  - `lin`
  - `etm`

- **Special Flags**
  - `labels` → supervised classification with labels
  - `authors` → supervised author-topic variant
  - `ecrtm` → adds ECR regularization

**Examples:**
- `scholar-labels-lin-dir_pathwise-etm`
- `scholar-lin-dir_rsvi-lin`
- `scholar-authors-lin-dir_pathwise-etm`

**Summary**
- **LDA** and **BERTopic**: classic baselines.
- **VAE**: highly configurable, supports unsupervised + context + LLM embeddings + labels/authors/ECR.
- **SCHOLAR**: supervised variant, label/author aware.

---

## Custom Datasets

If you provide a custom `dataset_name`, point `dataset_path` to a folder with:
```
train.csv
val.csv
test.csv
```
Each file should include columns:
- `text`  — the document string
- `label` — (optional) label string(s) for supervised models like SCHOLAR
- `author` — (optional) author id/name if using author-conditioned models

> Ensure that **all splits contain all labels** if you plan to use labels.

Built-in loaders also support:
- `20ng` via `sklearn.datasets`
- `ag_news` and `dbpedia` via `datasets` (Hugging Face)

---

## Evaluation

Use `tomo.evaluation`:
```python
from tomo.evaluation import return_coherence, return_topic_diversity

# topics is a list[List[str]]; text is tokenized docs for coherence
per_topic, overall = return_coherence(topics, text, metric="c_v")  # or "u_mass", "c_npmi"
diversity = return_topic_diversity(topics)
```

- **Coherence** uses `gensim.models.CoherenceModel`.
- **Diversity** is the ratio of unique words over total topic words.

---

## Model Notes

### LDA
- Backed by `gensim`. Provide `num_topics`. Preprocesses documents and fits LDA.

### BERTopic
- Uses `sentence-transformers` for doc embeddings and HDBSCAN + c-TF-IDF internally.

### VAE family
- Flexible encoder/sampler/decoder.
- Samplers: `dir_pathwise` (pathwise gradients), `dir_rsvi` (reparameterized RSVI).
- Decoders: `lin` (linear), `etm` (Embeddings Topic Model decoder).  
- Optional flags:
  - `context`: appends sentence-transformer document embeddings.
  - `llm`: expect external LLM embeddings (pass as `doc_emb` in data loaders).
  - `labels`, `authors`: adds supervised heads (multi-task NLL).

### SCHOLAR
- Supervised topic model with a classifier over latent topics. Use `labels` flag and provide label names.

---

## Programmatic Data Pipeline

```python
from tomo.data import return_prepared_data, return_dataloaders

prepared = return_prepared_data(dataset_name="20ng", device="cpu", remove_labels=False, tpl=1)
train_dl, val_dl, test_dl = return_dataloaders(prepared, batch_size=128, device="cpu")
```

`return_prepared_data` handles tokenization (via [`tokenease`](https://pypi.org/project/tokenease/)), vocabulary building, optional sentence-transformer doc embeddings, and label/index mappings.

---

## Logging

Set `wandb: true` to enable Weights & Biases logging. Artifacts (topics, metrics, runtime) are saved under `exp_path` (and in W&B if enabled).

---

## Reproducibility

Use `random_state` and set the `device` (e.g., `"cuda:0"`). You may also want to set:
```python
import torch, numpy as np, random
torch.manual_seed(0); np.random.seed(0); random.seed(0)
```

---

## 📖 How to Cite

If you use **tomo** in your research, please cite the relevant works:

### When using **VAE with labels and/or authors**
```bibtex
@inproceedings{nagda-etal-2025-tethering,
    title = "Tethering Broken Themes: Aligning Neural Topic Models with Labels and Authors",
    author = "Nagda, Mayank  and
      Ostheimer, Phil  and
      Fellenz, Sophie",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2025",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-naacl.44/",
    doi = "10.18653/v1/2025.findings-naacl.44",
    pages = "740--760",
    ISBN = "979-8-89176-195-7"
}
```
### When using RSVI sampler

```bibtex
@article{burkhardt2019decoupling,
  title={Decoupling sparsity and smoothness in the dirichlet variational autoencoder topic model},
  author={Burkhardt, Sophie and Kramer, Stefan},
  journal={Journal of Machine Learning Research},
  volume={20},
  number={131},
  pages={1--27},
  year={2019}
}
```

## License

This project is MIT licensed. See `LICENSE`.
