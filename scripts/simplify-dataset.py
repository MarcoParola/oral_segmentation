import argparse
import json
import os

# dict containing the class aggregation criterias
config = dict(
    neoplastic=dict(
        id=1,
        color="#fc0000",
        categories=["neoplastica"],
    ),
    aphthous=dict(
        id=2,
        color="#00fc00",
        categories=["ALU", "aftosa_maior", "aftosa"]
    ),
    traumatic=dict(
        id=3,
        color="#fcfcfc",
        categories=["traumatica_cronica", "traumatica_acuta"]
    )
)


def agregate_classes(json_dataset):
    new_categories = []
    new_categories_map = dict()
    for name, newcat in config.items():
        new_categories.append(dict(
            id=newcat["id"],
            name=name,
            supercategory="",
            color=newcat["color"],
            metadata=dict(),
            keypoint_colors=[]
        ))
        for oldcat_name in newcat["categories"]:
            for oldcat in json_dataset["categories"]:
                if oldcat["name"] == oldcat_name:
                    new_categories_map[oldcat["id"]] = newcat["id"]

    usable_images = []
    new_annotations = []
    for annotation in json_dataset["annotations"]:
        if annotation["category_id"] in new_categories_map.keys():
            annotation["category_id"] = new_categories_map[annotation["category_id"]]
            new_annotations.append(annotation)
            usable_images.append(annotation["image_id"])

    new_images = []
    for image in json_dataset["images"]:
        if image["id"] in usable_images:
            new_images.append(image)

    json.dump(dict(
        images=new_images,
        annotations=new_annotations,
        categories=new_categories
    ), open(os.path.join(args.folder, "dataset.json"), "w"), indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    args = parser.parse_args()
    dataset = json.load(open(os.path.join(args.folder, "coco_dataset.json"), "r"))
    agregate_classes(dataset)
    print("OK!")