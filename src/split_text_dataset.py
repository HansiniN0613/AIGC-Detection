import pandas as pd

from sklearn.model_selection import train_test_split


# ==================================================
# SETTINGS
# ==================================================

INPUT_FILE = "data/text/hc3_balanced.csv"

TRAIN_FILE = "data/text/train.csv"

VALIDATION_FILE = "data/text/validation.csv"

TEST_FILE = "data/text/test.csv"

RANDOM_STATE = 42


# ==================================================
# LOAD DATASET
# ==================================================

print()
print("**************************************")
print("TEXT DATASET SPLITTING")
print("**************************************")


df = pd.read_csv(
    INPUT_FILE
)


print()
print("Total samples:")
print(len(df))


# ==================================================
# FIRST SPLIT
# 70% TRAIN
# 30% TEMPORARY
# ==================================================

train_df, temporary_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)


# ==================================================
# SECOND SPLIT
# 15% VALIDATION
# 15% TEST
# ==================================================

validation_df, test_df = train_test_split(
    temporary_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temporary_df["label"]
)


# ==================================================
# SAVE DATASETS
# ==================================================

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

validation_df.to_csv(
    VALIDATION_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)


# ==================================================
# DISPLAY RESULTS
# ==================================================

print()
print("--------------------------------------")

print(
    "Training samples:",
    len(train_df)
)

print(
    "Validation samples:",
    len(validation_df)
)

print(
    "Testing samples:",
    len(test_df)
)


print()
print("Training label distribution:")

print(
    train_df["label"].value_counts()
)


print()
print("Validation label distribution:")

print(
    validation_df["label"].value_counts()
)


print()
print("Testing label distribution:")

print(
    test_df["label"].value_counts()
)


print()
print("**************************************")
print("DATASET SPLITTING COMPLETE")
print("**************************************")