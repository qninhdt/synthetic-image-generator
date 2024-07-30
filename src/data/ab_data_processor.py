import torch
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as TF
import lightning.pytorch.callbacks.model_checkpoint


class ABDataProcessor:
    def __init__(self, size):
        self.size = size

        self.image_transform = A.Compose(
            [
                A.Resize(size, size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ],
        )

    def preprocess(self, image):
        image = np.array(image)
        return self.image_transform(image=image)["image"]

    def preprocess_sample(self, sample):
        output_sample = dict()

        if "A_image" in sample:
            output_sample["A_image"] = self.preprocess(sample["A_image"])
            output_sample["A_size"] = sample["A_image"].size

        if "B_image" in sample:
            output_sample["B_image"] = self.preprocess(sample["B_image"])
            output_sample["B_size"] = sample["B_image"].size

        return output_sample

    def postprocess(self, images, original_sizes=None, to_pil=True):
        images = (images * 0.5) + 0.5

        output_images = []

        for i, image in enumerate(images):

            if original_sizes:
                image = TF.resize(image, original_sizes[i])

            if to_pil:
                image = TF.to_pil_image(image)
            else:
                image = (image * 255).type(torch.uint8)
                image = torch.clip(image, 0, 255)

            output_images.append(image)

        return output_images
