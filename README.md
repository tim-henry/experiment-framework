# experiment-framework

A framework designed to organize experiments. The idea is to modularize the experiment into 4 components;
the input dataset, the specific experiment to run, the model to run the experiment on, and the output function to process
the raw results. This allows for quick experimentation with new datasets or models.

### Usage
usage: main.py [-h] --input INPUT --output OUTPUT --model MODEL --experiment
               EXPERIMENT
               
All arguments are required and each must refer to an entry in its corresponding config file's "options" dict.
