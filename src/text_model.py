import torch
import torch.nn as nn
from transformers import RobertaModel


class TextAIDetector(nn.Module):
    """
    RoBERTa-based binary classifier for detecting
    human-written vs AI-generated text.

    Label:
        0 = Human
        1 = AI-generated
    """

    def __init__(self):
        super().__init__()

        self.roberta = RobertaModel.from_pretrained(
            "roberta-base"
        )

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(
            768,
            2
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # First token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        x = self.dropout(cls_embedding)

        logits = self.classifier(x)

        return logits