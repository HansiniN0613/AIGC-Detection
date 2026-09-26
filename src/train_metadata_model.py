import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from metadata_model import MetadataAIDetector


# ============================================================
# SETTINGS
# ============================================================

TRAIN_FILE = "data/text/train_metadata.csv"
VAL_FILE = "data/text/validation_metadata.csv"
TEST_FILE = "data/text/test_metadata.csv"

MODEL_FILE = "models/metadata_detector.pth"
SCALER_FILE = "models/metadata_scaler.pkl"


FEATURE_COLUMNS = [
    "text_length",
    "word_count",
    "sentence_count",
    "avg_word_length",
    "hashtag_count",
    "mention_count",
    "url_count",
    "emoji_count",
    "uppercase_ratio",
    "punctuation_ratio"
]


BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("METADATA MODEL TRAINING")
print("=" * 60)

print("Device:", device)


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

print()
print("Train samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))


# ============================================================
# SELECT FEATURES
# ============================================================

X_train = train_df[FEATURE_COLUMNS].values
y_train = train_df["label"].values

X_val = val_df[FEATURE_COLUMNS].values
y_val = val_df["label"].values

X_test = test_df[FEATURE_COLUMNS].values
y_test = test_df["label"].values


# ============================================================
# SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)

X_test = scaler.transform(X_test)


# Save scaler
joblib.dump(
    scaler,
    SCALER_FILE
)

print()
print("Scaler saved to:", SCALER_FILE)


# ============================================================
# CONVERT TO PYTORCH TENSORS
# ============================================================

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

model = MetadataAIDetector(
    input_size=len(FEATURE_COLUMNS)
)

model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    average_loss = total_loss / len(loader)

    return (
        average_loss,
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# TRAINING
# ============================================================

best_f1 = 0.0


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for features, labels in train_loader:

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    train_loss = total_loss / len(train_loader)

    (
        val_loss,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1
    ) = evaluate(
        model,
        val_loader
    )


    print()
    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    print(
        f"Val Precision: {val_precision:.4f}"
    )

    print(
        f"Val Recall: {val_recall:.4f}"
    )

    print(
        f"Val F1: {val_f1:.4f}"
    )


    # Save best model
    if val_f1 > best_f1:

        best_f1 = val_f1

        torch.save(
            model.state_dict(),
            MODEL_FILE
        )

        print(
            "Best metadata model saved."
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device
    )
)


# ============================================================
# FINAL TEST
# ============================================================

(
    test_loss,
    test_accuracy,
    test_precision,
    test_recall,
    test_f1
) = evaluate(
    model,
    test_loader
)


print()
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"Test Loss:      {test_loss:.4f}"
)

print(
    f"Test Accuracy:  {test_accuracy:.4f}"
)

print(
    f"Test Precision: {test_precision:.4f}"
)

print(
    f"Test Recall:    {test_recall:.4f}"
)

print(
    f"Test F1:        {test_f1:.4f}"
)

print()
print("Model saved to:")
print(MODEL_FILE)

print()
print("Scaler saved to:")
print(SCALER_FILE)

print()
print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)