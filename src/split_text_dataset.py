import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


# ======================================
# CONFIGURATION
# ======================================

INPUT_FILE = Path("data/text/hc3_multisource_raw.csv")
OUTPUT_DIR = Path("data/text")


# ======================================
# LOAD DATASET
# ======================================

print("\n======================================")
print("TEXT DATASET SPLITTING")
print("======================================\n")

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Total samples loaded: {len(df)}")


# ======================================
# CHECK REQUIRED COLUMNS
# ======================================

required_columns = ["text", "label", "source"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

print("\nRequired columns found:")
print(required_columns)


# ======================================
# REMOVE EMPTY TEXT
# ======================================

before = len(df)

df = df.dropna(subset=["text"])
df = df[df["text"].str.strip() != ""]

after = len(df)

print(f"\nEmpty texts removed: {before - after}")
print(f"Samples remaining: {after}")


# ======================================
# FIRST SPLIT
# TRAIN = 80%
# TEMP = 20%
# ======================================

print("\nCreating train/test split...")

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df[["label", "source"]]
)


# ======================================
# SECOND SPLIT
# VALIDATION = 10%
# TEST = 10%
# ======================================

print("Creating validation/test split...")

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df[["label", "source"]]
)


# ======================================
# RESET INDEX
# ======================================

train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ======================================
# SAVE DATASETS
# ======================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_file = OUTPUT_DIR / "train.csv"
validation_file = OUTPUT_DIR / "validation.csv"
test_file = OUTPUT_DIR / "test.csv"

train_df.to_csv(train_file, index=False)
validation_df.to_csv(validation_file, index=False)
test_df.to_csv(test_file, index=False)


# ======================================
# DISPLAY RESULTS
# ======================================

print("\n======================================")
print("DATASET SPLITTING COMPLETE")
print("======================================")

print(f"\nTrain samples:      {len(train_df)}")
print(f"Validation samples: {len(validation_df)}")
print(f"Test samples:       {len(test_df)}")

print("\n--------------------------------------")
print("TRAIN LABEL DISTRIBUTION")
print("--------------------------------------")
print(train_df["label"].value_counts())

print("\n--------------------------------------")
print("VALIDATION LABEL DISTRIBUTION")
print("--------------------------------------")
print(validation_df["label"].value_counts())

print("\n--------------------------------------")
print("TEST LABEL DISTRIBUTION")
print("--------------------------------------")
print(test_df["label"].value_counts())


print("\n--------------------------------------")
print("TRAIN SOURCE DISTRIBUTION")
print("--------------------------------------")
print(
    train_df.groupby(["source", "label"]).size()
)


print("\n--------------------------------------")
print("VALIDATION SOURCE DISTRIBUTION")
print("--------------------------------------")
print(
    validation_df.groupby(["source", "label"]).size()
)


print("\n--------------------------------------")
print("TEST SOURCE DISTRIBUTION")
print("--------------------------------------")
print(
    test_df.groupby(["source", "label"]).size()
)


print("\n--------------------------------------")
print("FILES SAVED")
print("--------------------------------------")

print(train_file)
print(validation_file)
print(test_file)

print("\nDone!")