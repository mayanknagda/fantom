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
