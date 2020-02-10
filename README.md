# HiggsML_Investigations
A sequel repo to QCHS-2018 for continuing testing deep learning methods on the HiggsML dataset

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
