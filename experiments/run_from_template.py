# This script demonstrates how to launch an experiment from a YAML config.
# Key steps:
#   1) Read the config (dataset, model, training hyperparameters).
#   2) Load & split data (optionally using registered datasets or your CSVs).
#   3) Build the model via `tomo.models.get_model(...)` using `model_name`.
#   4) Train & evaluate; logs and artifacts are written to `exp_path`.
# Tips:
#   - See `config/template.yaml` for all available fields.
#   - Use `wandb: true` to log losses/metrics; set `project_name` accordingly.
#   - For LDA, set `model_name: lda`; for BERTopic, `bertopic`.
#   - For VAE/SCHOLAR models, `model_name` is hyphenated like:
#       `vae-[context|llm|labels|authors]-[dir_pathwise|dir_rsvi]-[lin|etm]`
#     Examples:
#       - `vae-context-dir_rsvi-lin` (uses contextual doc embeddings).
#       - `scholar-labels-dir_rsvi-lin` (supervised SCHOLAR with labels).
#   - Set `num_topics` explicitly or leave `null` to infer from labels (where applicable).

# a simple script to run an experiment based on a config file. The config file is given as input from the command line. Just run python run_from_template.py /path/to/config.yaml
import sys
import os
import yaml
from tomo.run import run

template_path = sys.argv[1]

kwargs = yaml.safe_load(open(template_path, "r"))

if not os.path.exists(kwargs["exp_path"]):
    os.makedirs(kwargs["exp_path"])

with open(os.path.join(kwargs["exp_path"], "config.yaml"), "w") as outfile:
    yaml.dump(kwargs, outfile)

run(**kwargs)
