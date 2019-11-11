import gzip
import json
import os
from argparse import ArgumentParser
from collections import defaultdict

from tqdm import tqdm


def read_raw_dataset_games(file_path, successful_only=False):
    if file_path.endswith(".gz"):
        with gzip.open(file_path) as f:
            for line in f:
                line = line.decode("utf-8")
                game = json.loads(line.strip("\n"))

                if not successful_only or (successful_only and game["status"] == "success"):
                    yield game
    else:
        with open(file_path) as f:
            for line in f:
                game = json.loads(line.strip("\n"))

                if not successful_only or (successful_only and game["status"] == "success"):
                    yield game


guesswhat_files = {
    "train": "guesswhat.train.jsonl.gz",
    "valid": "guesswhat.valid.jsonl.gz",
    "test": "guesswhat.test.jsonl.gz"
}

parser = ArgumentParser()

parser.add_argument("-dataset_path", required=True, type=str, help="Path containing the GuessWhat?! dataset files")
parser.add_argument("-output_path", default="data/", type=str,
                    help="Path containing the GuessWhat?! object annotations in COCO format")


def main(args):
    image_annotations = {}

    for split_name, split_file in guesswhat_files.items():
        dataset_path = os.path.join(args.dataset_path, split_file)

        print("Reading dataset object information from file: {}".format(split_name))
        for game in tqdm(read_raw_dataset_games(dataset_path, successful_only=False)):
            image_id = game["image"]["id"]

            if image_id not in image_annotations:
                image_annotations[image_id] = {
                    "annotations": [],
                    "objects": set()
                }

            object_set = image_annotations[image_id]["objects"]

            for obj in game["objects"]:
                if obj["id"] not in object_set:
                    obj["image_id"] = image_id
                    object_set.add(obj["id"])
                    image_annotations[image_id]["annotations"].append(obj)

    output_file = os.path.join(args.output_path, "object_annotations.json")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    print("Writing bounding boxes info to file: {}".format(output_file))
    with open(output_file, mode="w") as out_file:
        json.dump({
            "annotations": {image_id: data["annotations"] for image_id, data in image_annotations.items()}
        }, out_file)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
