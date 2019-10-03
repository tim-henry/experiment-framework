import json
import os
import torch


# Config file containing output functions


def keep_pct_readout_dump(timestamp, output_values, dataset=None, modelname=None):
    with open("results/" + timestamp + '/train.json', 'w') as fp:
        json.dump(output_values["train_results"], fp)

    with open("results/" + timestamp + '/test.json', 'w') as fp:
        json.dump(output_values["test_results"], fp)

    if "state_dict" in output_values:
        directory = "analysis/state_dicts/" + dataset + "/" + modelname + "/"
        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
        for keep_pct in output_values["state_dict"].keys():
            torch.save(output_values["state_dict"][keep_pct],
                       directory + str(keep_pct) + ".pt")


options = {"keep_pct_readout_dump": keep_pct_readout_dump}
