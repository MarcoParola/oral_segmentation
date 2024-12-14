# **Documentation**

The project is composed of the following modules, more details are below:

- [Main scripts for training and test models](#main-scripts-for-training-and-test-models)
- [Pytorch-lightning modules (data and models)](#pytorch-lightning-modules)
- [Configuration handling](#configuration-handling)
- [Additional utility scripts](#additional-utility-scripts)


## Main scripts for training and test models

All the experiments consist of train and test classification and segmentation architectures. You can use `train.py` and `test.py`, respectively. 

TODO write a bit more (?)

## Pytorch-lightning modules
Since this project is developed using the `pytorch-lightning` framework, two key concepts are `Modules` to declare a new model and `DataModules` to organize of our dataset. Both of them are declared in `src/`, specifically in `src/models/` and `src/dataset.py`, respectively. More information are in the next sections.

### Deep learning models

In this project the following segmentation models are implemented:
- FCN
- DeepLab

TODO put a mini description

### Datasets and datamodules

TODO

## Configuration handling
The configuration managed with [Hydra](https://hydra.cc/). Every aspect of the configuration is located in `config/` folder. The file containing all the configuration is `config.yaml`; edit it to change any configuration parameters.

## Additional utility scripts

In the `scripts/` folder, there are all independent files not involved in the `pytorch-lightning` workflow for data preparation and visualization.

Due to the possibility of errors in the dataset, such as missing images, run the check-dataset.py script to detect such errors. Returns the elements to be removed from the json file (this can be done manually or via a script).
```bash
python -m scripts.check-dataset --dataset data/coco_dataset.json
```
In this work, the dataset was annotated with more labels than necessary. Some are grouped under more general labels. To aggregate all the labels of the three diseases studied in this work, the following script is executed. In addition, we split the dataset with the holdout method.
```bash
python -m scripts.simplify-dataset --folder data
python -m scripts.split-dataset --folder data
```

You can use the `dataset-stats.py`   script to print the class occurrences for each dataset.
```bash
python -m scripts.dataset-stats --dataset data/dataset.json # entire dataset
python -m scripts.dataset-stats --dataset data/train.json # training set
python -m scripts.dataset-stats --dataset data/test.json # test set
```
