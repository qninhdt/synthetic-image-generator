import os
import numpy as np
from PIL import Image
import torch
import json
from random import randint, shuffle
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from tqdm import tqdm


class BDD100KDataset(Dataset):

    def __init__(self, path, size, split, paired=True):
        self.path = path
        self.size = size
        self.split = split
        self.paired = paired

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

        A_images = []
        B_images = []

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
                    A_images.append(label)
                elif timeofday == "night":
                    B_images.append(label)

        self.annotations = A_images + B_images

        self.n_A_images = len(A_images)
        self.n_B_images = len(B_images)

        print(f"Loaded {self.n_A_images} daytime images")
        print(f"Loaded {self.n_B_images} nighttime images")

    def __len__(self):
        if self.paired:
            return max(self.n_A_images, self.n_B_images)
        else:
            return self.n_A_images + self.n_B_images

    def load_image(self, name):
        return Image.open(
            os.path.join(self.path, "images", self.size, self.split, name)
        ).convert("RGB")

    def __getitem__(self, idx):
        if self.paired:
            A = self.annotations[idx % self.n_A_images]
            # randomly select a B image to avoid fixed pairs
            B = self.annotations[self.n_A_images + randint(0, self.n_B_images - 1)]

            A_image = self.load_image(A["name"])
            B_image = self.load_image(B["name"])

            return {
                "A_image": A_image,
                "B_image": B_image,
                "A_size": A_image.size,
                "B_size": B_image.size,
            }
        else:
            label = self.annotations[idx]
            image = self.load_image(label["name"])

            if idx < self.n_A_images:
                return {
                    "A_image": image,
                    "A_size": image.size,
                }
            else:
                return {
                    "B_image": image,
                    "B_size": image.size,
                }
