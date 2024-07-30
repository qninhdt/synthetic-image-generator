from typing import List

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


def concat_images(images: List[Image.Image] | List[torch.Tensor]) -> Image.Image:
    """Concatenates a list of images horizontally.

    :param images: A list of images to concatenate.

    :return: The concatenated image.
    """
    if isinstance(images[0], torch.Tensor):
        images = [TF.to_pil_image(image) for image in images]

    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im
