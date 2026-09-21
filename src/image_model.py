import torch
import torch.nn as nn

from transformers import ViTModel


class ImageAIDetector(nn.Module):
    """
    ViT-based binary classifier for detecting
    human-created vs AI-generated images.

    Label:
        0 = Human
        1 = AI-generated
    """

    def __init__(self):

        super().__init__()

        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224"
        )

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(
            768,
            2
        )

    def forward(self, pixel_values):

        outputs = self.vit(
            pixel_values=pixel_values
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]

        x = self.dropout(
            cls_embedding
        )

        logits = self.classifier(x)

        return logits