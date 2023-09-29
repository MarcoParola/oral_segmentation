import argparse
import json
import random
import os

def most_frequent(arr): 
    return max(set(arr), key = arr.count) 

def create_coco(images):
    global images_ids
    global dataset
    coco_images = []
    coco_annotations = []
    
    for image in images:
        coco_images.append(images_ids[image["id"]])
        for annotation in image["annotations"]:
            coco_annotations.append(annotation)
    
    return dict(
        images=coco_images,
        annotations=coco_annotations,
        categories=dataset["categories"]
    )


parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True)
parser.add_argument("--train-perc", type=float, default=0.7)
parser.add_argument("--val-perc", type=float, default=0.15)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

dataset = json.load(open(os.path.join(args.folder, "dataset.json"), "r"))

category_buckets = dict()
category_names = dict()
images_ids = dict()
images = dict()

for image in dataset["images"]:
    images_ids[image["id"]] = image

for category in dataset["categories"]:
    category_names[category["id"]] = category["name"]

for annotation in dataset["annotations"]:
    if annotation["image_id"] not in images:
        images[annotation["image_id"]] = dict(annotations=[], c=None, id=annotation["image_id"])
    images[annotation["image_id"]]["annotations"].append(annotation)

for id, image in images.items():
    categories = list(map(lambda a: a["category_id"], image["annotations"]))
    image["c"] = most_frequent(categories)

for id, image in images.items():
    if image["c"] not in category_buckets:
        category_buckets[image["c"]] = []
    category_buckets[image["c"]].append(image)


print("Categories:")

train_images = []
test_images = []
val_images = []
manual_categories = []

for key, elems in category_buckets.items():
    print("- %s: %d" % (category_names[key], len(elems)))
    if category_names[key] in manual_categories:
        for image in elems:
            if images_ids[image["id"]]["file_name"] in manual[category_names[key]]["test"]:
                print("Adding %s to test" % (images_ids[image["id"]]["file_name"]))
                test_images.append(image)
            else:
                print("Adding %s to train" % (images_ids[image["id"]]["file_name"]))
                train_images.append(image)
    else:
        random.shuffle(elems)
        pivot1 = int(len(elems) * args.train_perc)
        pivot2 = int(len(elems) * args.val_perc) + pivot1
        train_images.extend(elems[:pivot1])
        val_images.extend(elems[pivot1:pivot2])
        test_images.extend(elems[pivot2:])


json.dump(create_coco(train_images), open(os.path.join(args.folder, "train.json"), "w"), indent=2)
json.dump(create_coco(val_images), open(os.path.join(args.folder, "val.json"), "w"), indent=2)
json.dump(create_coco(test_images), open(os.path.join(args.folder, "test.json"), "w"), indent=2)
print("OK!")


