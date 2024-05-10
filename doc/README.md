# **Documentation**

The project is composed of the following modules, more details are below:

- [Main scripts for training and test models](#main-scripts-for-training-and-test-models)
- [Pytorch-lightning modules (data and models)](#pytorch-lightning-modules)
- [Configuration handling](#configuration-handling)
- [Additional utility scripts](#additional-utility-scripts)


## Main scripts for training and test models

All the experiments consist of train and test classification and segmentation architectures. You can use `train.py` and `test.py`, respectively. 
`train.py` accepts the training and validation datasets as inputs. The model learns to identify patterns and relationships within the data by adjusting its parameters. During training, it is possible to visualize various metrics to understand their trends over the epochs. After training, the computed weights are saved into checkpoints, which can then be used by `test.py`.
`test.py` accepts the test dataset and a saved checkpoint as inputs. It prints the final metrics to the command line and saves the segmentation predictions in the directory `photo_output/version_{nameCheckpoint}/`. The output includes a plot containing three subfigures: the original image, the actual segmentation, and the predicted segmentation.

## Pytorch-lightning modules
Since this project is developed using the `pytorch-lightning` framework, two key concepts are `Modules` to declare a new model and `DataModules` to organize of our dataset. Both of them are declared in `src/`, specifically in `src/models/` and `src/dataset.py`, respectively. More information are in the next sections.

### Deep learning models

In this project the following segmentation models are implemented:
- FCN (Fully Convolutional Network): introduced to address the limitations faced by traditional Convolutional Neural Networks (CNNs) in the context of image segmentation to classify each individual pixel within an image.
- DeepLab: extends and improves upon the concept of FCNs, introducing innovative techniques to better address the problem of semantic segmentation.
- Unet: The architecture differs significantly from FCN and DeepLab. In this new architecture, a U-shaped scheme is leveraged, where the network is composed of two parts: a section called downsampling, which follows the typical architecture of a convolutional network, and an upsampling section. 

### Datasets and datamodules

The dataset contains photographic documentation from medical examinations conducted by the Oral Medicine Unit of the P. Giaccone University Hospital in Palermo, Italy, between 2021 and 2024. This image collection focuses on three distinct pathologies: neoplastic, aphthous, and traumatic. An annotation process was carried out using the COCO Annotator tool, with which lesions visible in the photographs were manually labeled by a dedicated dental team. For each image collected, the segment annotations and the pathology ID are maintained.

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
You can use the `threshold-optimization.py` and `threshold-optimization-ensemble.py` script to find the best threshold to use to binarize the output.
```bash
python -m scripts.threshold-optimization checkpoints.version={number/name}
python -m threshold-optimization-ensemble.py 
```
