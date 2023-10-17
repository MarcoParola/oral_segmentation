## How execute: python -m scripts.dataset-stats --dataset data\train.json --type all
## How execute: python -m scripts.dataset-stats --dataset data\train.json --type class

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import json

def plot_distribution_allData(json_data):
    id_area = dict()
    conta=0
    for annotation in json_data["annotations"]:
        #find width and height
        for image in json_data['images']:
            if image['id'] == annotation["image_id"]:
                width = image['width']
                height = image['height']
                file_name = image['file_name']
                break
        if annotation["image_id"] not in id_area:
            id_area[annotation["image_id"]] = 0  
        id_area[annotation["image_id"]] += (annotation["area"]/(width*height))*100     # if an img has more then one mask sum them

        """
        # Stampa alcune immagini in base alla percentuale del segmento estratto
        if id_area[annotation["image_id"]]<10 and conta<3 :
            conta+=1
            immagine = mpimg.imread("data/oral1/"+file_name)
            plt.imshow(immagine)
            plt.axis('off')
            plt.show()
        """

    plt.hist(id_area.values(), bins=50, color='blue', edgecolor='black')
    plt.title('Istogramma delle quantità di bianco nelle maschere di segmentazione')
    plt.xlabel('Quantità di "1" (pixel bianchi)')
    plt.ylabel('Frequenza')
    plt.show()

def plot_distribution_perClass(json_data):
    id_area_cat1 = dict()
    id_area_cat2 = dict()
    id_area_cat3 = dict()
    for annotation in json_data["annotations"]:
        if annotation["category_id"] == 1:
            #find width and height
            for image in json_data['images']:
                if image['id'] == annotation["image_id"]:
                    width = image['width']
                    height = image['height']
                    break
            if annotation["image_id"] not in id_area_cat1:
                id_area_cat1[annotation["image_id"]] = 0  
            id_area_cat1[annotation["image_id"]] += (annotation["area"]/(width*height))*100
        elif annotation["category_id"] == 2:
            #find width and height
            for image in json_data['images']:
                if image['id'] == annotation["image_id"]:
                    width = image['width']
                    height = image['height']
                    break
            if annotation["image_id"] not in id_area_cat2:
                id_area_cat2[annotation["image_id"]] = 0  
            id_area_cat2[annotation["image_id"]] += (annotation["area"]/(width*height))*100
        elif annotation["category_id"] == 3:
            #find width and height
            for image in json_data['images']:
                if image['id'] == annotation["image_id"]:
                    width = image['width']
                    height = image['height']
                    break
            if annotation["image_id"] not in id_area_cat3:
                id_area_cat3[annotation["image_id"]] = 0  
            id_area_cat3[annotation["image_id"]] += (annotation["area"]/(width*height))*100     # if an img has more then one mask sum them

    plt.hist(id_area_cat1.values(), bins=100, color='blue', edgecolor='black')
    plt.title('Categoria 1: Istogramma delle quantità di bianco nelle maschere di segmentazione')
    plt.xlabel('Quantità di "1" (pixel bianchi)')
    plt.ylabel('Frequenza')
    plt.show()

    plt.hist(id_area_cat2.values(), bins=100, color='blue', edgecolor='black')
    plt.title('Categoria 2: Istogramma delle quantità di bianco nelle maschere di segmentazione')
    plt.xlabel('Quantità di "1" (pixel bianchi)')
    plt.ylabel('Frequenza')
    plt.show()

    plt.hist(id_area_cat3.values(), bins=100, color='blue', edgecolor='black')
    plt.title('Categoria 3: Istogramma delle quantità di bianco nelle maschere di segmentazione')
    plt.xlabel('Quantità di "1" (pixel bianchi)')
    plt.ylabel('Frequenza')
    plt.show()


if __name__ == '__main__':
    '''script to check for missing image files
    expects a cli parameter "--dataset" valued by the json file path of the coco dataset
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--type", type=str, required=True)
    args = parser.parse_args()
    dataset = json.load(open(args.dataset, "r"))
    if args.type=="all":
        plot_distribution_allData(dataset)
    elif args.type=="class":
        plot_distribution_perClass(dataset)
    else:
        print("Error: option type not correct")
    