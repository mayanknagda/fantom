import torch
import yaml
import pickle
import numpy as np
from tomo.run import run
from tomo.data import return_prepared_data, return_dataloaders
from tomo.models import get_model
from sentence_transformers import SentenceTransformer

modelst = SentenceTransformer("all-MiniLM-L6-v2")

kwargs = yaml.safe_load(
    open(
        "/Users/mayank/Documents/projects/topic models/topic-models/experiments/hparams.yaml"
    )
)
# get data only once
prepared_data = return_prepared_data(**kwargs)
print("Using Topic Model")
topic_train_dl, topic_val_dl, topic_test_dl = return_dataloaders(
    prepared_data=prepared_data, **kwargs
)
kwargs["saved_data"] = (topic_train_dl, topic_val_dl, topic_test_dl, prepared_data)
# train the topic model
run(**kwargs)
# load the topic model
kwargs["i2w"] = {v: k for k, v in prepared_data["vocab"].items()}
kwargs["num_topics"] = len(prepared_data["label_to_topic"])
kwargs["vocab_embeddings"] = prepared_data["vocab_embeddings"]
topic_model = get_model(**kwargs)
# load the best model
topic_model.load_state_dict(torch.load(kwargs["exp_path"] + "/model.pt"))
topic_model.eval()
# get labels for the documents
kwargs["model_name"] += "_labels"
topic_train_dl, topic_val_dl, topic_test_dl = return_dataloaders(
    prepared_data=prepared_data, **kwargs
)
# load topic embeddings
topic_embeddings = pickle.load(open(kwargs["exp_path"] + "/topic_embeddings.pkl", "rb"))
topic_embeddings = torch.from_numpy(topic_embeddings).float()


def dl_to_emb(dl):
    doc_emb = []
    labels = []
    for batch in dl:
        en_out = topic_model.encoder(batch["bow"].to(kwargs["device"]))
        en_out = torch.nn.functional.softmax(en_out, dim=-1)
        doc_emb.append(torch.matmul(en_out, topic_embeddings).detach().cpu().numpy())
        labels.append(batch["labels"].detach().cpu().numpy().argmax(axis=-1))
    labels = np.concatenate(labels)
    doc_emb = np.concatenate(doc_emb)
    return doc_emb, labels


train_doc_emb, train_labels = dl_to_emb(topic_train_dl)
val_doc_emb, val_labels = dl_to_emb(topic_val_dl)
test_doc_emb, test_labels = dl_to_emb(topic_test_dl)


class Data(torch.utils.data.Dataset):
    def __init__(self, doc_emb, labels):
        self.doc_emb = doc_emb
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.doc_emb[idx], self.labels[idx]


train_dl, val_dl, test_dl = (
    torch.utils.data.DataLoader(
        Data(train_doc_emb, train_labels), batch_size=32, shuffle=True
    ),
    torch.utils.data.DataLoader(
        Data(val_doc_emb, val_labels), batch_size=32, shuffle=True
    ),
    torch.utils.data.DataLoader(
        Data(test_doc_emb, test_labels), batch_size=32, shuffle=True
    ),
)


class ClassificationModel(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_dim):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_dim, out_features=out_features),
        )

    def forward(self, x):
        return self.fc(x)


model = ClassificationModel(in_features=300, out_features=20, hidden_dim=64)
opt = torch.optim.AdamW(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()
model.to(kwargs["device"])
best_val_acc = 0
for epoch in range(100):
    for batch in train_dl:
        opt.zero_grad()
        out = model(batch[0].to(kwargs["device"]))
        loss = loss_fn(out, batch[1].to(kwargs["device"]))
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_acc = []
        for batch in train_dl:
            out = model(batch[0].to(kwargs["device"]))
            train_acc.append(
                (out.argmax(dim=-1) == batch[1].to(kwargs["device"])).float().mean()
            )
        val_acc = []
        for batch in val_dl:
            out = model(batch[0].to(kwargs["device"]))
            val_acc.append(
                (out.argmax(dim=-1) == batch[1].to(kwargs["device"])).float().mean()
            )
        print(
            f"Epoch: {epoch} | Train Acc: {torch.tensor(train_acc).mean().item()} | Val Acc: {torch.tensor(val_acc).mean().item()}"
        )
        if torch.tensor(val_acc).mean().item() > best_val_acc:
            best_val_acc = torch.tensor(val_acc).mean().item()
            torch.save(model.state_dict(), kwargs["exp_path"] + "/cl_topic_model.pt")
model.load_state_dict(torch.load(kwargs["exp_path"] + "/cl_topic_model.pt"))
with torch.no_grad():
    test_acc = []
    for batch in test_dl:
        out = model(batch[0].to(kwargs["device"]))
        test_acc.append(
            (out.argmax(dim=-1) == batch[1].to(kwargs["device"])).float().mean()
        )
    print(f"Test Acc Using Topic embeddings: {torch.tensor(test_acc).mean().item()}")

print("Using Sentence Transformers")
train_text = prepared_data["train_text"]
val_text = prepared_data["val_text"]
test_text = prepared_data["test_text"]
train_labels = prepared_data["train_labels"].argmax(axis=-1)
val_labels = prepared_data["val_labels"].argmax(axis=-1)
test_labels = prepared_data["test_labels"].argmax(axis=-1)
train_emb = modelst.encode(train_text)
val_emb = modelst.encode(val_text)
test_emb = modelst.encode(test_text)
print("Classification using Sentence Transformers")

train_dl, val_dl, test_dl = (
    torch.utils.data.DataLoader(
        Data(train_emb, train_labels), batch_size=32, shuffle=True
    ),
    torch.utils.data.DataLoader(Data(val_emb, val_labels), batch_size=32, shuffle=True),
    torch.utils.data.DataLoader(
        Data(test_emb, test_labels), batch_size=32, shuffle=True
    ),
)

model = ClassificationModel(in_features=384, out_features=20, hidden_dim=64)
opt = torch.optim.AdamW(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()
model.to(kwargs["device"])
best_val_acc = 0
for epoch in range(100):
    for batch in train_dl:
        opt.zero_grad()
        out = model(batch[0].to(kwargs["device"]))
        loss = loss_fn(out, batch[1].to(kwargs["device"]))
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_acc = []
        for batch in train_dl:
            out = model(batch[0].to(kwargs["device"]))
            train_acc.append(
                (out.argmax(dim=-1) == batch[1].to(kwargs["device"])).float().mean()
            )
        val_acc = []
        for batch in val_dl:
            out = model(batch[0].to(kwargs["device"]))
            val_acc.append(
                (out.argmax(dim=-1) == batch[1].to(kwargs["device"])).float().mean()
            )
        print(
            f"Epoch: {epoch} | Train Acc: {torch.tensor(train_acc).mean().item()} | Val Acc: {torch.tensor(val_acc).mean().item()}"
        )
        if torch.tensor(val_acc).mean().item() > best_val_acc:
            best_val_acc = torch.tensor(val_acc).mean().item()
            torch.save(model.state_dict(), kwargs["exp_path"] + "/cl_st_model.pt")
model.load_state_dict(torch.load(kwargs["exp_path"] + "/cl_st_model.pt"))
with torch.no_grad():
    test_acc = []
    for batch in test_dl:
        out = model(batch[0].to(kwargs["device"]))
        test_acc.append(
            (out.argmax(dim=-1) == batch[1].to(kwargs["device"])).float().mean()
        )
    print(
        f"Test Acc Using Sentence Transformers: {torch.tensor(test_acc).mean().item()}"
    )
