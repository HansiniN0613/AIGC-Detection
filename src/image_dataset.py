import os

import torch

from torch.utils.data import Dataset

from PIL import Image

from transformers import ViTImageProcessor


class ImageDataset(Dataset):

    def __init__(
        self,
        image_folder,
        max_images=None
    ):

        self.image_folder = image_folder

        self.processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )

        self.image_paths = []

        for filename in os.listdir(image_folder):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp")
            ):

                self.image_paths.append(
                    os.path.join(
                        image_folder,
                        filename
                    )
                )

        if max_images is not None:

            self.image_paths = self.image_paths[
                :max_images
            ]

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        encoding = self.processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = encoding[
            "pixel_values"
        ].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "image_path": image_path
        }