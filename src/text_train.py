import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW

from text_model import TextAIDetector
from text_dataset import TextDataset


# Settings

DATASET_PATH = "../data/text/sample_text.csv"

BATCH_SIZE = 2

EPOCHS = 1

LEARNING_RATE = 2e-5

MODEL_SAVE_PATH = "../models/text_detector.pth"


# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Dataset

dataset = TextDataset(
    DATASET_PATH
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


print("Number of training samples:", len(dataset))


# Model

model = TextAIDetector()

model = model.to(device)


# Optimizer

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# Loss function

criterion = torch.nn.CrossEntropyLoss()


# Training

model.train()


for epoch in range(EPOCHS):

    total_loss = 0

    for batch in dataloader:

        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch[
            "label"
        ].to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids,
            attention_mask
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss / len(dataloader)
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {average_loss:.4f}"
    )


# Save model

torch.save(
    model.state_dict(),
    MODEL_SAVE_PATH
)

print(
    "Model saved to:",
    MODEL_SAVE_PATH
)