# Attribute Combinations
#### Timothy Henry, Jamell Dozier

## Getting Started

### 1. Data Synthesis
1. Run "util/save_transformed.py" with the dataset of interest uncommented.<br/>
2. The aforementioned script may fail if the dataset relies on no digit 9 examples being present.
 If this is the case, use "util/remove_9s.ipynb" to modify the downloaded 
 raw dataset before rerunning "util/data_synthesis/main.py".

### 2. Training
usage: python main.py [-h] --input INPUT --model MODEL --experiment
               EXPERIMENT
               
example: python main.py --input colored_mnist --model resnet18 --experiment controlled_bias_training
               
All arguments are required and each must refer to an entry in its corresponding "config/" file's "options" dict.

### 3. Analysis
- #### Behavioral
1. In "util/results_plotting.ipynb", 
fill the desired experiment parameters in and run
- #### Neural 
##### - Activations
1. Generate dataset for analysis using "util/save_dataset.ipynb"
2. TODO(Jamell)
##### - Selectivity / Invariance
1. Generate dataset for selectivity and invariance calculation using "util/invariance_data.ipynb"
2. TODO(Jamell)

## Additional Usages
### 1. Loss Weight Tuning
Use "util/results_plotting_loss_weight_tuning
### 2. Dataset Visualization
Use the notebooks within "util/data_visualization"
### 3. Combination Space Subdivision
Use "util/generalized combinations.ipynb"
### 4. Finding Raw Experiment Output Files
Use "util/db.ipynb" to filter through and get timestamps and directories for experiments under some criteria
