import pandas as pd
import numpy as np
import json
import os
import yaml

with open("params.yaml", "r") as file:
    data = yaml.safe_load(file)

    metrics_data = {
        "choice": data["choice"]["properties"]["whalrus_rule"],
        "recommender_weight": data["choice"]["properties"]["recommender_weight"],
    }

    json_filename = "eval/metrics.json"
    with open(json_filename, "w") as json_file:
        json.dump(metrics_data, json_file, indent=4)
