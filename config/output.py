import json


# Config file containing output functions


def keep_pct_readout_dump(timestamp, output_values):
    with open("results/" + timestamp + '/train.json', 'w') as fp:
        json.dump(output_values["train_results"], fp)

    with open("results/" + timestamp + '/test.json', 'w') as fp:
        json.dump(output_values["test_results"], fp)


options = {"keep_pct_readout_dump": keep_pct_readout_dump}
