import argparse
from datetime import datetime
from os import mkdir
import sqlite3
import config.input
import config.output
import config.model
import config.experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='the name of the desired input data configuration')
    parser.add_argument('--output', type=str, required=True, help='the name of the desired output data configuration')
    parser.add_argument('--model', type=str, required=True, help='the name of the desired model configuration')
    parser.add_argument('--experiment', type=str, required=True, help='the name of the desired experiment config')
    args = parser.parse_args()
    print("Args:\nExperiment: {}\nOutput: {}\nInput: {}\nModel: {}".format(
        args.experiment, args.output, args.input, args.model))

    # Get input loader functions
    train_loader_fn, test_loader_fn = config.input.options[args.input]

    # Get output function
    output_fn = config.output.options[args.output]

    # Get model function
    model_fn = config.model.options[args.model]

    # Get experiment
    experiment = config.experiment.options[args.experiment]

    # Run the experiment
    timestamp = datetime.utcnow().strftime('%Y-%m-%d_%H:%M:%S_%f')
    output_values = experiment(train_loader_fn, test_loader_fn, model_fn)
    mkdir("results/" + timestamp)
    output_fn(timestamp, output_values, args.input, args.model, args.experiment)

    # Make a respective entry in the results lookup table
    while True:  # Try to obtain lock
        try:
            conn = sqlite3.connect('results/results_lookup.db')
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS  experiments
                         (timestamp text, input text, output text, model text, experiment text)''')
            c.execute("INSERT INTO experiments VALUES (?,?,?,?,?)",
                      (timestamp, args.input, args.output, args.model, args.experiment))
            conn.commit()
            conn.close()
        except:
            continue
        break


if __name__ == '__main__':
    main()
