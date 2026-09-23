import os

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import ViTImageProcessor


class ImageDataset(Dataset):
    """
    Dataset for loading images and preparing them
    for the ViT image model.

    Label:
        0 = Human / real image
        1 = AI-generated image
    """

    def __init__(
        self,
        image_folder,
        label=None,
        max_images=None
    ):

        self.image_folder = image_folder
        self.label = label

        self.processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )

        self.image_paths = []

        if not os.path.exists(image_folder):
            raise FileNotFoundError(
                f"Image folder not found: {image_folder}"
            )

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

        self.image_paths.sort()

        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

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

        sample = {
            "pixel_values": pixel_values,
            "image_path": image_path
        }

        if self.label is not None:
            sample["label"] = torch.tensor(
                self.label,
                dtype=torch.long
            )

        return sample