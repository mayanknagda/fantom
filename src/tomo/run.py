import os
import torch
import pickle
import json
import wandb
from time import time
from .data import return_prepared_data, return_dataloaders
from .models import get_model
from .optim import get_trainer
from .evaluation import return_coherence, return_topic_diversity


def run(**kwargs):
    # wandb init
    if kwargs["wandb"] is False:
        os.environ["WANDB_MODE"] = "offline"
    run = wandb.init(
        project=kwargs["project_name"],
        name=kwargs["model_name"],
        config=kwargs,
        dir=kwargs["wandb_path"],
    )
    # create experiment path if it does not exist.
    if not os.path.exists(kwargs["exp_path"]):
        os.makedirs(kwargs["exp_path"])
    else:
        # if the experiment path exists, just check if there are logs already stored at this location.
        if os.path.exists(os.path.join(kwargs["exp_path"], "summary.pkl")):
            print("Experiment already exists! Untrained model will be returned soon.")
            return None, None
    # get data if not there
    if kwargs["saved_data"] is None:
        # if data is not saved. jthen prepare data from scratch. Otherwise just load it.
        prepared_data = return_prepared_data(**kwargs)
        train_dl, val_dl, test_dl = return_dataloaders(
            prepared_data=prepared_data, **kwargs
        )
    else:
        train_dl, val_dl, test_dl, prepared_data = kwargs["saved_data"]
    # build a int to word mapping.
    kwargs["i2w"] = {v: k for k, v in prepared_data["vocab"].items()}
    if kwargs["num_topics"] is None:
        # if number of topics are given as null just use total number of labels present. Make sure that the data is labeled if you wanna use num_topics as None.
        kwargs["num_topics"] = len(prepared_data["label_names"])
    if "authors" in kwargs["model_name"]:
        # if the model has anything to do with authors (author model), get the author vocab and build a int to author mapping.
        kwargs["i2a"] = prepared_data["authors_vocab"]
    if "labels" or "scholar" in kwargs["model_name"]:
        kwargs["label_names"] = prepared_data["label_names"]
    if "etm" in kwargs["model_name"]:
        # if the model is an ETM, we will need pre-trained vocab embeddings as part of the decoder.
        kwargs["vocab_embeddings"] = prepared_data["vocab_embeddings"]
    # dump kwargs now
    with open(os.path.join(kwargs["exp_path"], "kwargs.pkl"), "wb") as f:
        pickle.dump(kwargs, f)
    # make model and run
    if ("lda" in kwargs["model_name"]) or ("bertopic" in kwargs["model_name"]):
        if "lda" in kwargs["model_name"]:
            model = get_model(train_texts=prepared_data["train_wl_text"], **kwargs)
            summary = {}
            start_training_time = time()
            model.run()
            end_training_time = time()
            time_taken = end_training_time - start_training_time
        else:
            model = get_model(train_texts=prepared_data["train_text"], **kwargs)
            summary = {}
            start_training_time = time()
            model.run()
            end_training_time = time()
            time_taken = end_training_time - start_training_time
    else:
        model = get_model(**kwargs)
        model.to(kwargs["device"])
        # optimize
        opt = torch.optim.Adam(model.parameters(), lr=kwargs["lr"])
        trainer = get_trainer(
            model=model,
            train_dl=train_dl,
            val_dl=val_dl,
            optimizer=opt,
            **kwargs,
        )
        start_training_time = time()
        trainer.run(epochs=kwargs["epochs"])
        end_training_time = time()
        time_taken = end_training_time - start_training_time
        # logging
        summary = trainer.summary
        # load best model
        model.load_state_dict(torch.load(os.path.join(kwargs["exp_path"], "model.pt")))
    summary["time_taken"] = time_taken
    # get files to save from the model
    files_to_save = model.files_to_save()
    # iterate through the files and save them.
    for file_name, file in files_to_save.items():
        if file_name.endswith(".json"):
            if file_name == "topics.json":
                try:
                    # eval topics.json (coherence and diversity on train set)
                    text = prepared_data["text"]
                    topic_words = file["topic_words"]
                    c_v = return_coherence(topic_words, text, "c_v")
                    c_npmi = return_coherence(topic_words, text, "c_npmi")
                    div = return_topic_diversity(topic_words)
                    file["c_v_per_topic"] = c_v[0]
                    file["c_v_avg"] = c_v[1]
                    file["c_npmi_per_topic"] = c_npmi[0]
                    file["c_npmi_avg"] = c_npmi[1]
                    file["diversity"] = div
                    # wandb log
                    run.summary["c_v"] = c_v[1]
                    run.summary["c_npmi"] = c_npmi[1]
                    run.summary["diversity"] = div
                except:
                    pass
            with open(os.path.join(kwargs["exp_path"], file_name), "w") as f:
                json.dump(file, f)
            # add topics.json to wandb
            if file_name == "topics.json":
                wandb_run_path = run.dir
                with open(os.path.join(wandb_run_path, file_name), "w") as f:
                    json.dump(file, f)
        else:
            with open(os.path.join(kwargs["exp_path"], file_name), "wb") as f:
                pickle.dump(file, f)
    # save summary
    with open(os.path.join(kwargs["exp_path"], "summary.pkl"), "wb") as f:
        pickle.dump(summary, f)
    # wandb log
    run.summary["time_taken"] = time_taken
    run.finish()
    # cuda memory management
    torch.cuda.empty_cache()
    return model, summary
