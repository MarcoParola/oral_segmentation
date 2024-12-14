# **POCI-dataset**

[![license](https://img.shields.io/github/license/MarcoParola/POCI-dataset?style=plastic)]()
[![size](https://img.shields.io/github/languages/code-size/MarcoParola/POCI-dataset?style=plastic)]()

**WORK IN PROGRESS** dataset will be officially released after paper acceptance.

This GitHub repo is twinned with releasing the multi-purpose Photographic Oral Cancer Imaging (POCI)** dataset supporting open oral cancer research.


## Install 

To create a new the project working on the POCI-dataset, clone the repository and install dependencies:
```sh
git clone https://github.com/MarcoParola/POCI-dataset.git
cd POCI-dataset
```

Create the virtualenv (you can also use conda) and install the dependencies of *requirements.txt*

```sh
python -m venv env
env/Scripts/activate
python -m pip install -r requirements.txt
mkdir data
```

Download the oral POCI-dataset (both images and json file) from **TODO-put-kaggle-link**. Copy them into `data` folder and unzip the file `oral1.zip`.

## Utilities

You can find torch datasets for several computer vision datasets on `src/datasets/`, including classification, segmentation, etc.

You can use the `dataset-stats.py`   script to print the class occurrences for each dataset.
```
python -m scripts.dataset-stats --dataset data\dataset.json # entire dataset
python -m scripts.dataset-stats --dataset data\train.json # training set
python -m scripts.dataset-stats --dataset data\test.json # test set
```

The configuration is managed with [Hydra](https://hydra.cc/). Every aspect of the configuration is located in `config/` folder. The file containing all the configuration is `config.yaml`; edit it to change any configuration parameters.
