import argparse
import json

def compute_stats(json_data):
    category_count = dict()
    for annotation in json_data["annotations"]:
        if annotation["category_id"] not in category_count:
            category_count[annotation["category_id"]] = 0
        category_count[annotation["category_id"]] += 1

    for category in json_data["categories"]:
        print("-%s: %d" % (category["name"], category_count[category["id"]]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    dataset = json.load(open(args.dataset, "r"))
    compute_stats(dataset)
