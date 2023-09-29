import argparse
import json
import os

def check_missing_images(json_dataset):
    images = dict()
    for image in json_dataset["images"]:
        image_path = os.path.join(os.path.dirname(args.dataset), "oral1", image["file_name"])
        if not os.path.exists(image_path):
            print("Missing image (in images):", image_path)
        images[image["id"]] = image

    for annotation in json_dataset["annotations"]:
        if annotation["image_id"] not in images:
            print("Missing image ID (in annotations):", annotation["image_id"])
            continue
        image = images[annotation["image_id"]]
        image_path = os.path.join(os.path.dirname(args.dataset), "oral1", image["file_name"])
        if not os.path.exists(image_path):
            print("Missing image (in annotations):", image_path)


if __name__ == '__main__':
    '''script to check for missing image files
    expects a cli parameter "--dataset" valued by the json file path of the coco dataset
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    dataset = json.load(open(args.dataset, "r"))
    check_missing_images(dataset)

