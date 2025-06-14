# Topic Models (tomo package)

## Installation

Create a dedicated conda/python environment. Manually install PyTorch according to your system specifications. Subsequently, install Poetry within this environment. From the root folder, execute `poetry install` to install the tomo package in your environment, enabling you to develop your experimental scripts.

To facilitate running experiments on your custom dataset, a pre-existing script is available:

## Execute on Your Dataset:

Begin by crafting your configuration file adhering to the provided template located at `config/template.yaml`. The data preparation process is elucidated within the template.yaml file. Execute the experiment based on a specific configuration using the following command: `python experiments/run_from_template.py path/to/config.yaml`.

> For further details, consult the README.md files within the package for additional documentation.