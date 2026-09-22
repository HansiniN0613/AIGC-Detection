import pandas as pd
import numpy as np
import torch

from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# ======================================
# CONFIGURATION
# ======================================

MODEL_DIR = Path("models/text_roberta")
TEST_FILE = Path("data/text/test.csv")

MAX_LENGTH = 256


# ======================================
# DEVICE
# ======================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("\n======================================")
print("TEXT MODEL EVALUATION")
print("======================================")

print(f"\nDevice: {device}")


# ======================================
# LOAD TEST DATA
# ======================================

print("\nLoading test dataset...")

test_df = pd.read_csv(TEST_FILE)

print(f"Test samples: {len(test_df)}")

print("\nTest label distribution:")
print(test_df["label"].value_counts())


# ======================================
# KEEP REQUIRED COLUMNS
# ======================================

test_df = test_df[
    ["text", "label", "source"]
].copy()


# ======================================
# CONVERT TO DATASET
# ======================================

test_dataset = Dataset.from_pandas(
    test_df,
    preserve_index=False
)


# ======================================
# LOAD TOKENIZER
# ======================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)


# ======================================
# TOKENIZE
# ======================================

def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


print("Tokenizing test dataset...")

test_tokenized = test_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing test data"
)


# ======================================
# LOAD TRAINED MODEL
# ======================================

print("\nLoading trained model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)


# ======================================
# DATA COLLATOR
# ======================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ======================================
# TRAINER
# ======================================

trainer = Trainer(
    model=model,
    data_collator=data_collator,
)


# ======================================
# PREDICTIONS
# ======================================

print("\n======================================")
print("GENERATING TEST PREDICTIONS")
print("======================================")

predictions = trainer.predict(
    test_tokenized
)


logits = predictions.predictions

predicted_labels = np.argmax(
    logits,
    axis=1
)

true_labels = np.array(
    test_df["label"]
)


# ======================================
# OVERALL METRICS
# ======================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels,
    predicted_labels,
    average="binary",
    zero_division=0
)


print("\n======================================")
print("OVERALL TEST RESULTS")
print("======================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ======================================
# CLASSIFICATION REPORT
# ======================================

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=[
            "Human",
            "AI-generated"
        ],
        digits=4
    )
)


# ======================================
# CONFUSION MATRIX
# ======================================

print("\n======================================")
print("CONFUSION MATRIX")
print("======================================")

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print(cm)


# ======================================
# SOURCE-WISE EVALUATION
# ======================================

print("\n======================================")
print("SOURCE-WISE RESULTS")
print("======================================")

results = []

for source in sorted(
    test_df["source"].unique()
):

    mask = (
        test_df["source"] == source
    )

    source_true = true_labels[mask]

    source_pred = predicted_labels[mask]

    source_accuracy = accuracy_score(
        source_true,
        source_pred
    )

    source_precision, source_recall, source_f1, _ = (
        precision_recall_fscore_support(
            source_true,
            source_pred,
            average="binary",
            zero_division=0
        )
    )

    results.append({
        "source": source,
        "samples": len(source_true),
        "accuracy": source_accuracy,
        "precision": source_precision,
        "recall": source_recall,
        "f1": source_f1
    })


results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False
    )
)


# ======================================
# SAVE SOURCE RESULTS
# ======================================

output_file = Path(
    "data/text/text_model_test_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print(
    f"\nSource-wise results saved to:\n{output_file}"
)


print("\n======================================")
print("EVALUATION COMPLETE")
print("======================================")