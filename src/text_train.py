import os
import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW

from text_model import TextAIDetector
from text_dataset import TextDataset


# -------------------------------------------------
# SETTINGS

DATASET_PATH = "data/text/sample_text.csv"

BATCH_SIZE = 2

EPOCHS = 1

LEARNING_RATE = 2e-5

MODEL_SAVE_PATH = "models/text_detector.pth"


# --------------------------------------------------
# DEVICE

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("--------------------------------------")
print("TEXT DETECTOR TRAINING")
print("--------------------------------------")

print("Using device:", device)


# --------------------------------------------------
# DATASET

print()
print("Loading dataset...")

dataset = TextDataset(
    DATASET_PATH
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(
    "Number of samples:",
    len(dataset)
)


# --------------------------------------------------
# MODEL

print()
print("Loading RoBERTa model...")

model = TextAIDetector()

model = model.to(device)


# --------------------------------------------------
# OPTIMIZER

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# LOSS FUNCTION

criterion = torch.nn.CrossEntropyLoss()


# --------------------------------------------------
# TRAINING

print()
print("Starting training...")

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

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(
            input_ids,
            attention_mask
        )

        # Calculate loss
        loss = criterion(
            logits,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update model
        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss / len(dataloader)
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {average_loss:.4f}"
    )


# --------------------------------------------------
# SAVE MODEL

os.makedirs(
    "models",
    exist_ok=True
)

torch.save(
    model.state_dict(),
    MODEL_SAVE_PATH
)

print()
print("--------------------------------------")
print("TRAINING COMPLETE")
print("--------------------------------------")

print(
    "Model saved to:",
    MODEL_SAVE_PATH
)