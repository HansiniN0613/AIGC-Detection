import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import RobertaTokenizer


class TextDataset(Dataset):

    def __init__(
        self,
        csv_file,
        max_length=512
    ):

        self.data = pd.read_csv(csv_file)

        self.tokenizer = RobertaTokenizer.from_pretrained(
            "roberta-base"
        )

        self.max_length = max_length

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        text = str(
            self.data.iloc[index]["text"]
        )

        label = int(
            self.data.iloc[index]["label"]
        )

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),

            "attention_mask": encoding[
                "attention_mask"
            ].squeeze(0),

            "label": torch.tensor(
                label,
                dtype=torch.long
            )
        }