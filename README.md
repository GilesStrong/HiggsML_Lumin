# On the impact of modern deep-learning techniques to the performance and time-requirements of classification models in experimental high-energy physics

This repo is designed to support the paper "On the impact of modern deep-learning techniques to the performance and time-requirements of classification models in experimental high-energy physics", Strong, 2019, [arXiv:2002.01427 [physics.data-an]](https://arxiv.org/abs/2002.01427). It contains code to rerun the experiments performed in the paper to reproduce the results, allow users to better understand how each method is used, and provide a baseline for future comparisons.

## Installation

### Get code & data

1. `git clone https://github.com/GilesStrong/HiggsML_Lumin.git`
1. `cd HiggsML_Lumin`
1. `mkdir data`
1. `wget -O data/atlas-higgs-challenge-2014-v2.csv.gz http://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz`
1. `gunzip data/atlas-higgs-challenge-2014-v2.csv.gz`

### Install requirements

#### Via PIP

- `pip install -r requirements.txt`

#### Via Conda environment

 1. `conda env create -f environment.yml`
 1. `conda activate higgsml_lumin`

## Running

The experiments are run using [Jupyter Notebooks](https://jupyter.org/), which can be accessed by running:

- `jupyter notebook`

In the browser window which should have opened, navigate to the `notebooks` directory. Here there are several directories and four notebooks. Each directory is associated with a different experiment and contains all the notebooks relevant to that particular experiment.

### Running experiments

Each directory contains a single notebook which can be duplicated to run multiple times and save the results. Experiment 13 (`13_swish_ensemble_embed_aug_onecycle_dense`), which was the final model used for the paper, contains an example of this where the same experiment was run six different times on six different computing setups.

Rerunning of the experiments also uses different random seeds for splitting of the validation data, as described in the paper. This is achieved by configuring the third cell in the notebooks, which contains `experiment = Experiment(NAME, 'mbp', RESULTS_PATH)`. Where `NAME` is the basic of the experiment, e.g. `'13_swish_ensemble_embed_aug_onecycle_dense'`, and `'mbp'` is the name of the computing setup. These names are used to lookup particular settings in the `Experiment` class, defined in `./modules/basics.py`, where each machine is assigned its own random seed, as well as a description which is used later for annotating plots and results. When `Experiment.save()` is called, the results are written to `./results/{experiment name}_{machine name}.json'`.

Users should edit `Experiment` in `./modules/basics.py` to include their own machines and names. Each notebook is designed to be run top-to-bottom, except for those in `17_hyperparam_search` which will be discussed later.

### Comparing results

`notebooks/Results_Viewing.ipynb` takes experiment results from `./results` and compares average performance between configurations. The variable `BLIND` determines whether the private AMS results should be shown to the user. By default this is `True` to attempt to preserve challenge conditions. It is recommended to only set this to `False` once you are happy with your model configuration.

Results are loaded using the `Result` class located in ./modules/basics.py`, which loads up the results and computes mean values for the metrics, and also has functions for comparing configurations and producing plots.

The git repo currently ships with single results for each experiment, except for the final model (experiment 13) where six example results are available. In order to reproduce the results of the paper, one should run each experiment several more times, to get average results.

### Hyper-parameter scan

The last experiment (17) is the hyper-parameter scan used to try to find a better architecture. The notebook included when run will sample parameters and train an ensemble of three networks. This repeats 30 times. Results are saved between each iteration and past results are automatically loaded. This notebook should ideally be run on several machines simultaneously, which may need restarting from time to time according to memory requirements.

Once sufficient results have been collected, `notebooks/17_hyperparam_search/Analysis.ipynb` can be used to analyse them, fit the Gaussian processes, and discover promising new architectures. It is left to the user to configure new experiments to run these new architectures, if they so wish.

### Misc. notebooks

- `notebooks/Results_Viewing.ipynb` contains code to plot out and view features from the data
- `notebooks/Feature_Selection.ipynb` runs the feature selection tests used for the paper
- `notebooks/Feature_Selection-Full.ipynb` runs a more advanced set of feature selection tests which were not used for the paper, but might of interest to the reader

## Citation

Please cite as:

@misc{strong2020impact,  
  author        = {Giles Chatham Strong},  
  title         = {On the impact of modern deep-learning techniques to the performance and time-requirements of classification models in experimental high-energy physics},  
  year          = 2020,  
eprint        =   {2002.01427},  
archivePrefix =   {arXiv},  
primaryClass  =   {physics.data-an}  
}
