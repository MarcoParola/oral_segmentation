# **Oral segmentation**

[![license](https://img.shields.io/github/license/MarcoParola/oral_segmentation?style=plastic)]()
[![size](https://img.shields.io/github/languages/code-size/MarcoParola/oral_segmentation?style=plastic)]()

This GitHub repo is to publicly release the code of [Oral cancer recognition on photographic images via deep learning semantic segmentation (SSCI 2025)](https://ieeexplore.ieee.org/document/11002690). Here is a quick guide on how to install and use the repo. More information is in the [official documentation](doc/README.md).

![example](https://github.com/MarcoParola/oral_segmentation/assets/32603898/8dc53d9c-6288-4b8e-a029-fa141c31ecc1)

Please cite the following:
```
@INPROCEEDINGS{11002690,
  author={Parola, Marco and Cimino, Mario G.C.A. and Cantini, Irene and Mantia, Gaetano La and Campisi, Giuseppina and Di Fede, Olga},
  booktitle={2025 IEEE Symposium on Computational Intelligence in Health and Medicine Companion (CIHM Companion)}, 
  title={Oral Cancer Recognition on Photographic Images Via Deep Learning Semantic Segmentation}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Deep learning;Pathology;Image recognition;Semantic segmentation;Computational modeling;Computer architecture;Predictive models;Lesions;Medical diagnostic imaging;Cancer;Oral cancer;Oral squamous cell carcinoma;Unet;Healthcare screening;Ensemble learning;Semantic segmentation},
  doi={10.1109/CIHMCompanion65205.2025.11002690}
}

@article{PAROLA2024102433,
  title = {Towards explainable oral cancer recognition: Screening on imperfect images via Informed Deep Learning and Case-Based Reasoning},
  journal = {Computerized Medical Imaging and Graphics},
  volume = {117},
  pages = {102433},
  year = {2024},
  issn = {0895-6111},
  doi = {https://doi.org/10.1016/j.compmedimag.2024.102433},
  url = {https://www.sciencedirect.com/science/article/pii/S0895611124001101},
  author = {Marco Parola and Federico A. Galatolo and Gaetano {La Mantia} and Mario G.C.A. Cimino and Giuseppina Campisi and Olga {Di Fede}},
  keywords = {Oral cancer, Oncology, Medical imaging, Case-based reasoning, Informed deep learning, Explainable artificial intelligence},
}
```

## Install 

To install the project, clone the repository and install dependencies:
```sh
git clone https://github.com/MarcoParola/oral_segmentation.git
cd oral_segmentation
```

Create the virtualenv (you can also use conda) and install the dependencies of *requirements.txt*

```sh
python -m venv env
env/Scripts/activate
python -m pip install -r requirements.txt
mkdir data
```


## Download dataset
Download the oral coco-dataset (both images and json file) from TODO-put-link. Copy them into `data` folder and unzip the file `oral1.zip`.

## Usage
Regarding the usage of this repo, in order to reproduce the experiments, we organize the workflow in two part: (i) data preparation and (ii) deep learning experiments.

### Data preparation
Due to the possibility of errors in the dataset, such as missing images, run the check-dataset.py script to detect such errors. Returns the elements to be removed from the json file (this can be done manually or via a script).
```
python -m scripts.check-dataset --dataset data\coco_dataset.json
```
In this work, the dataset was annotated with more labels than necessary. Some are grouped under more general labels. To aggregate all the labels of the three diseases studied in this work, the following script is executed. In addition, we split the dataset with the holdout method.
```
python -m scripts.simplify-dataset --folder data
python -m scripts.split-dataset --folder data
```

You can use the `dataset-stats.py`   script to print the class occurrences for each dataset.
```
python -m scripts.dataset-stats --dataset data\dataset.json # entire dataset
python -m scripts.dataset-stats --dataset data\train.json # training set
python -m scripts.dataset-stats --dataset data\test.json # test set
```


### Train
The training can be done using the implemented models (DeepLab, Fcn, and Unet). To launch it use the following commands.
```
python train.py model.model_type={networkName}
```
Network name options are (default=fcn):
```
- fcn 
- deeplab 
- unet 
```
To log metrics on tensorboard add (default=false):
```
log.tensorboard=True 
```
Only for unet, replace encoderName with efficientnet-b7 or resnet50 add (default=efficientnet-b7):
```
model.encoder_name='encoderName' 
```
For multiclass train add (default=1):
```
model.num_classes=3 
```


### Test single network
The test can recover any train version thanks to the saved checkpoints. The checkpoints have to be placed into `./logs/oral/`.

Pretrained weights can be downloaded [here](https://drive.google.com/file/d/1jRZuxER9ESNEsWZJdqha3k57fhvPYwv7/view?usp=drive_link) both for testing and ensemble experiments. 
Unzip and copy the checkpoint folders in `./logs/oral/`.

```
python test.py checkpoints.version={version number/network_name}
```
Network name options are: 
```
Binary train:
- fcn_bin
- deeplab_bin
- unet_eff_bin
- unet_res_bin 

Multiclass train:
- fcn_mul
- deeplab_mul
- unet_eff_mul
- unet_res_mul
```

To change the threshold in binary case add (default 0.5):
```
model.sgm_threshold={number} 
```


### Test ensemble
This type of test is able to recover 4 different checkpoints.
```
python testEnsemble.py
```
To change results aggregation add (default dec_fus=median, type_aggr=soft, num_classes=1):
```
ensemble.dec_fus={decisionFunction} ensemble.type_aggr={hard/soft} model.num_classes={1/3} 
```
To change the default checkpoints in binary case add:
```
ensemble.check_fcn_bin={name/number} ensemble.check_dl_bin={name/number} ensemble.check_unet_eff_bin={name/number} ensemble.check_unet_res_bin={name/number}
```
To change the default checkpoints in multiclass case add:
```
ensemble.check_fcn_mul={name/number} ensemble.check_dl_mul={name/number} ensemble.check_unet_eff_mul={name/number} ensemble.check_unet_res_mul={name/number}
```
Decision function options are:
```
- median
- mean
- max
- min
- product

- mv # for majority voting only with ensemble.type_aggr=hard

- weight # for weighted max only in multiclass case and ensemble.type_aggr=soft
```


### View logs and results on tensorboard

To view the tensorboard log start the server with the following command and connect to `localhost:6006`
```
python -m tensorboard.main --logdir=logs
```
