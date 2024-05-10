## How execute: python -m scripts.plot_sizeDistribution --dataset data\train.json 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import json

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

    
    plt.hist(id_area_cat1.values(), bins=100, color="blue", alpha=.3)
    plt.title('Category ID 1: Histogram of White Area Proportions in Segmentation Masks')
    plt.xlabel('Percentage')
    plt.ylabel('Frequency')
    plt.show()

    plt.hist(id_area_cat2.values(), bins=100, color="orange", alpha=.3)
    plt.title('Category ID 2: Histogram of White Area Proportions in Segmentation Masks')
    plt.xlabel('Percentage')
    plt.ylabel('Frequency')
    plt.show()

    plt.hist(id_area_cat3.values(), bins=100, color="green", alpha=.3)
    plt.title('Category ID 3: Histogram of White Area Proportions in Segmentation Masks')
    plt.xlabel('Percentage')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
    # Plot with all classes

    plt.hist(id_area_cat1.values(), bins=100, alpha=.3, label='Cat 1')
    plt.xlabel('Percentage')
    plt.ylabel('Frequency')
    #plt.show()

    plt.hist(id_area_cat2.values(), bins=100, alpha=.3, label='Cat 2')
    plt.xlabel('Percentage')
    plt.ylabel('Frequency')
    #plt.show()

    plt.hist(id_area_cat3.values(), bins=100, alpha=.3, label='Cat 3')
    plt.title('Histogram of White Area Proportions in Segmentation Masks per category')
    plt.xlabel('Percentage')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    '''script to check for missing image files
    expects a cli parameter "--dataset" valued by the json file path of the coco dataset
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    try:
        with open(args.dataset, "r") as file:
            dataset = json.load(file)
            plot_distribution_perClass(dataset)
    except FileNotFoundError:
        print(f"Error: File '{args.dataset}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{args.dataset}': {e}")
    