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

### Training
La parte di **data preparation** la puoi saltare, perchè ti ho gia fornito tutti i dati preprocessati e puliti, quindi puoi iniziare a guardare da qui. 

Per ora il training è molto facile, intanto riesegui il training con un modello e visualizza i risultati euristicamente con i plot simili a quelli che ti ho mandato in chat.
```
python train.py
```

# TODO
Mini lista guida delle prossime cose da fare, da prendere come linea guida e non come assolutismo; approfondisci tutti gli aspetti che trovi più interessanti. Dedica il tempo che meglio credi, dando priorità agli esami:
- Scarica i dati che ti ho messo sulla cartella su gdrive. I dati li mettiamo in un cartalla `data/` contenetenente i tre file json splittati e una cartella `oral1/` da unzippare contenente le immagini. Lancia lo script `dataset-stats.py` per vedere che tutto torni, è per visualizzare le classi (vedi sezione precedente)
- Rieseguire il codice del training e visualizzare i risultati euristicamente tramite plot. Considera che ho implementato due modelli, fare un po' di training di benchmark vari. Inoltre fare un po' di ricerche online sul layer finale che appendiamo a fine dei modelli (classifier[4]). 
- parametrizzare la scelta del modello, altrimenti ogni volta tocca mettere mano al codice e modificarlo manualmente. 
- studiare in letteratura quali funzioni di loss è meglio usare per fare il training di questi modelli
- studiare alcune metriche in letteratura per misurare le performance della segmentazione (consigli IoU, dice coefficient.. sicuramente ne esistono altre) e implementarle o prenderle pronte da qualche librerie.
- dopo averle implementate introdure il calcolo di queste metriche durante la fase di testing del modello
- fare i log su tensorboard (per questo riaggiorniamoci fra un po' che magari si fa insieme in dipartimento)
- implementare altri modelli pretrained (almeno un altro, magari altri due)
- valutare di fare la segmentazione per classi (questa cosa la facciamo dopo che hai eseguito più modelli sul dataset e abbiamo calcolato le performance)

NB. tutte le modifiche che fai sul codice, non farle sul branch `main`, ma su `develop` che ho appena creato, ogni volta che arriviamo ad una versione stabile, facciamo la merge sul main, ok? Ti lascio sto [link](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) se hai bisogno di vedere qualcosa in più su git
