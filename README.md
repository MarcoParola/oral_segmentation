# **Oral segmentation**

[![license](https://img.shields.io/github/license/MarcoParola/oral_segmentation?style=plastic)]()
[![size](https://img.shields.io/github/languages/code-size/MarcoParola/oral_segmentation?style=plastic)]()

This github repo is to publicly release the code of oral segmentation.


## Install

Create the virtualenv (you can also use conda) and install the dependencies of *requirements.txt*

```
python -m venv env
env/Scripts/activate
python -m pip install -r requirements.txt
mkdir data
```

If you download more libs, freeze them in the requirement file:
```
pyhton -m pip freeze > requirements.txt
```
Then you can download the oral coco-dataset (both images and json file) from TODO-put-link. Copy them into `data` folder and unzip the file `oral1.zip`.

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

### Experiments
La parte di **data preparation** la puoi saltare, perchè ti ho gia fornito tutti i dati preprocessati e puliti, quindi puoi iniziare a guardare da qui. 

## Train
Il train può essere fatto utilizzando entrambi i modelli implementati (DeepLab e Fcn). Per lanciarlo usare i seguenti comandi.
-train binario:
```
python train.py 
```
-train multiclasse:
```
python train.py model.num_classes=3 (Non ancora implementato)
```

## Test
Il test è in grado di recuperare l'ultimo train eseguito o una qualsiasi versione precedente grazie ai checkpoint salvati.
Per eseguire l'ultima versione:
```
python test.py
```
Per eseguire una versione specifica:
```
python test.py checkpoints.version= {numero}
```
Per salvare immagini soft:
```
python test.py model.sgm_type=soft
```



Per visualizzare i log di tensorboard avviare il server con il seguente comando e collegarsi a `localhost:6006`
```
python -m tensorboard.main --logdir=logs
```
Per ora sono visualizzabili solo i log del train e le metriche finali del test. Manca da testare il funzionamento del plot delle metriche durante il train.

# TODO
Mini lista guida delle prossime cose da fare, da prendere come linea guida e non come assolutismo; approfondisci tutti gli aspetti che trovi più interessanti. Dedica il tempo che meglio credi:

Documento tesi:
- crea un nuovo documento latex su [overleaf](https://it.overleaf.com) per la tesi e condividimelo marco.parola@ing.unipi.it 
- impostare le macrosezioni della tesi: introduction, background, state of the art 
- riguardo allo state dell'arte: fare uno studio sui vari problemi di segmentazione (NB sui problemi di segmentazione, non sulle tecniche): 1. semantic segmentation, 2. instance segmentation, 3. panoptic segmenatation. 
- valutare di fare la segmentazione per classi (questa cosa la facciamo dopo che hai eseguito più modelli sul dataset e abbiamo calcolato le performance)
- parametrizzare la scelta del modello, altrimenti ogni volta tocca mettere mano al codice e modificarlo manualmente. 
- implementare una versione di UNET


NB. tutte le modifiche che fai sul codice, non farle sul branch `main`, ma su `develop` che ho appena creato, ogni volta che arriviamo ad una versione stabile, facciamo la merge sul main
