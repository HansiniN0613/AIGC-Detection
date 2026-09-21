import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from tqdm import tqdm

from text_model import TextAIDetector
from text_dataset import TextDataset


# ======================================
# SETTINGS
# ======================================

TRAIN_FILE = "data/text/train.csv"
VALIDATION_FILE = "data/text/validation.csv"

BATCH_SIZE = 2
EPOCHS = 2
LEARNING_RATE = 2e-5

MODEL_SAVE_PATH = "models/text_detector_best.pth"


# ======================================
# DEVICE
# ======================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("======================================")
print("TEXT DETECTOR TRAINING")
print("======================================")
print()
print("Using device:", device)


# ======================================
# LOAD DATASETS
# ======================================

print()
print("Loading training dataset...")

train_dataset = TextDataset(TRAIN_FILE)

print("Training samples:", len(train_dataset))


print()
print("Loading validation dataset...")

validation_dataset = TextDataset(VALIDATION_FILE)

print("Validation samples:", len(validation_dataset))


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ======================================
# LOAD MODEL
# ======================================

print()
print("Loading RoBERTa model...")

model = TextAIDetector()
model = model.to(device)


# ======================================
# OPTIMIZER AND LOSS
# ======================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = torch.nn.CrossEntropyLoss()


# ======================================
# BEST MODEL
# ======================================

best_f1 = 0.0


# ======================================
# TRAINING
# ======================================

print()
print("======================================")
print("STARTING TRAINING")
print("======================================")


for epoch in range(EPOCHS):

    print()
    print("--------------------------------------")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("--------------------------------------")

    # ==================================
    # TRAINING
    # ==================================

    model.train()

    total_training_loss = 0

    progress_bar = tqdm(
        train_loader,
        desc="Training",
        unit="batch"
    )

    for batch in progress_bar:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

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

        total_training_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_training_loss = (
        total_training_loss / len(train_loader)
    )


    # ==================================
    # VALIDATION
    # ==================================

    print()
    print("Running validation...")

    model.eval()

    total_validation_loss = 0

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for batch in tqdm(
            validation_loader,
            desc="Validation",
            unit="batch"
        ):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(
                input_ids,
                attention_mask
            )

            loss = criterion(
                logits,
                labels
            )

            total_validation_loss += loss.item()

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities[:, 1].cpu().numpy()
            )


    average_validation_loss = (
        total_validation_loss /
        len(validation_loader)
    )


    # ==================================
    # METRICS
    # ==================================

    validation_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    validation_precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    validation_recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    validation_f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    validation_auc = roc_auc_score(
        all_labels,
        all_probabilities
    )


    # ==================================
    # DISPLAY RESULTS
    # ==================================

    print()
    print("======================================")
    print("VALIDATION RESULTS")
    print("======================================")

    print(
        f"Training Loss:       "
        f"{average_training_loss:.4f}"
    )

    print(
        f"Validation Loss:     "
        f"{average_validation_loss:.4f}"
    )

    print(
        f"Accuracy:             "
        f"{validation_accuracy * 100:.2f}%"
    )

    print(
        f"Precision:            "
        f"{validation_precision * 100:.2f}%"
    )

    print(
        f"Recall:               "
        f"{validation_recall * 100:.2f}%"
    )

    print(
        f"F1 Score:             "
        f"{validation_f1 * 100:.2f}%"
    )

    print(
        f"ROC-AUC:              "
        f"{validation_auc:.4f}"
    )


    # ==================================
    # SAVE BEST MODEL
    # ==================================

    if validation_f1 > best_f1:

        best_f1 = validation_f1

        os.makedirs(
            "models",
            exist_ok=True
        )

        torch.save(
            model.state_dict(),
            MODEL_SAVE_PATH
        )

        print()
        print("New best model saved!")

        print(
            f"Best Validation F1: "
            f"{best_f1 * 100:.2f}%"
        )


# ======================================
# COMPLETE
# ======================================

print()
print("======================================")
print("TRAINING COMPLETE")
print("======================================")

print()
print(
    f"Best Validation F1: "
    f"{best_f1 * 100:.2f}%"
)

print()
print("Model saved to:")
print(MODEL_SAVE_PATH)