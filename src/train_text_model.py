import pandas as pd
import numpy as np
import torch

from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


# ======================================
# CONFIGURATION
# ======================================

MODEL_NAME = "roberta-base"

TRAIN_FILE = Path("data/text/train.csv")
VALIDATION_FILE = Path("data/text/validation.csv")

MODEL_OUTPUT_DIR = Path("models/text_roberta")

MAX_LENGTH = 256


# ======================================
# DEVICE
# ======================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("\n======================================")
print("TEXT MODEL TRAINING")
print("======================================")

print(f"\nDevice: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available. Training will use CPU.")


# ======================================
# LOAD DATA
# ======================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

print(f"Training samples:   {len(train_df)}")
print(f"Validation samples: {len(validation_df)}")


# ======================================
# KEEP REQUIRED COLUMNS
# ======================================

train_df = train_df[["text", "label"]].copy()
validation_df = validation_df[["text", "label"]].copy()


# ======================================
# CONVERT TO HUGGING FACE DATASETS
# ======================================

train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False
)

validation_dataset = Dataset.from_pandas(
    validation_df,
    preserve_index=False
)


# ======================================
# LOAD TOKENIZER
# ======================================

print("\nLoading RoBERTa tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ======================================
# TOKENIZATION
# ======================================

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


print("Tokenizing training dataset...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing train data"
)

print("Tokenizing validation dataset...")

validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing validation data"
)


# ======================================
# LOAD MODEL
# ======================================

print("\nLoading RoBERTa classification model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={
        0: "Human",
        1: "AI-generated"
    },
    label2id={
        "Human": 0,
        "AI-generated": 1
    },
)


# ======================================
# DATA COLLATOR
# ======================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ======================================
# EVALUATION METRICS
# ======================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(
        predictions,
        axis=1
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ======================================
# TRAINING CONFIGURATION
# ======================================

training_args = TrainingArguments(
    output_dir=str(MODEL_OUTPUT_DIR),

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    num_train_epochs=1,

    weight_decay=0.01,

    logging_steps=100,

    load_best_model_at_end=True,
    metric_for_best_model="f1",

    save_total_limit=2,

    report_to="none",

    fp16=False,
)


# ======================================
# TRAINER
# ======================================

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=validation_dataset,

    processing_class=tokenizer,
    data_collator=data_collator,

    compute_metrics=compute_metrics,
)


# ======================================
# START TRAINING
# ======================================

print("\n======================================")
print("STARTING TRAINING")
print("======================================\n")

trainer.train()


# ======================================
# SAVE FINAL MODEL
# ======================================

print("\n======================================")
print("SAVING MODEL")
print("======================================")

MODEL_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

trainer.save_model(
    str(MODEL_OUTPUT_DIR)
)

tokenizer.save_pretrained(
    str(MODEL_OUTPUT_DIR)
)


# ======================================
# FINAL VALIDATION
# ======================================

print("\n======================================")
print("FINAL VALIDATION RESULTS")
print("======================================")

results = trainer.evaluate()

for key, value in results.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


print("\n======================================")
print("TRAINING COMPLETE")
print("======================================")

print(
    f"\nModel saved to:\n{MODEL_OUTPUT_DIR}"
)