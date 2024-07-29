import os
import numpy as np
from PIL import Image
import torch
import json
from random import randint
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from tqdm import tqdm


class BDD100KDataset(Dataset):

    def __init__(self, path, size, split):
        self.path = path
        self.size = size
        self.split = split

        self.annotation_path = os.path.join(path, "labels", f"det_{split}.txt")

        self.load_annotations()

    def load_annotations(self):

        # load all images id can be used
        image_ids = []

        for file in os.listdir(
            os.path.join(self.path, "images", self.size, self.split)
        ):
            if file.endswith(".jpg"):
                image_ids.append(file)

        image_ids = set(image_ids)

        self.A_annotations = []
        self.B_annotations = []

        with open(
            os.path.join(self.path, "labels", "det_20", f"det_{self.split}.json")
        ) as f:
            annotations = json.load(f)

            for image_label in tqdm(
                annotations, desc=f"Loading BDD100K {self.size} {self.split}"
            ):
                if image_label["name"] not in image_ids:
                    continue

                timeofday = image_label["attributes"]["timeofday"]

                label = {
                    "name": image_label["name"],
                }

                if timeofday == "daytime":
                    self.A_annotations.append(label)
                elif timeofday == "night":
                    self.B_annotations.append(label)

        print(f"Loaded {len(self.A_annotations)} daytime images")
        print(f"Loaded {len(self.B_annotations)} nighttime images")

    def __len__(self):
        return max(len(self.A_annotations), len(self.B_annotations))

    def load_image(self, name):
        return Image.open(
            os.path.join(self.path, "images", self.size, self.split, name)
        ).convert("RGB")

    def __getitem__(self, idx):
        A_idx = self.A_annotations[idx % len(self.A_annotations)]
        # randomly select a B image to avoid fixed pairs
        B_idx = self.B_annotations[randint(0, len(self.B_annotations) - 1)]

        A_image = self.load_image(A_idx["name"])
        B_image = self.load_image(B_idx["name"])

        return {
            "A_image": A_image,
            "B_image": B_image,
            "A_size": A_image.size,
            "B_size": B_image.size,
        }
