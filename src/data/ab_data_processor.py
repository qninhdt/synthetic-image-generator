import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as TF


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
        return {
            "A_image": self.preprocess(sample["A_image"]),
            "B_image": self.preprocess(sample["B_image"]),
            "A_size": sample["A_size"],
            "B_size": sample["B_size"],
        }

    def postprocess(self, image, original_size=None):
        image = (image * 0.5) + 0.5

        if original_size is not None:
            image = TF.resize(image, (original_size[1], original_size[0]))

        image = TF.to_pil_image(image)

        return image
