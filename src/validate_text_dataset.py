import pandas as pd


# ==================================================
# SETTINGS
# ==================================================

TRAIN_FILE = "data/text/train.csv"

VALIDATION_FILE = "data/text/validation.csv"

TEST_FILE = "data/text/test.csv"


# ==================================================
# LOAD DATASETS
# ==================================================

print()
print("======================================")
print("TEXT DATASET VALIDATION")
print("======================================")


train_df = pd.read_csv(
    TRAIN_FILE
)

validation_df = pd.read_csv(
    VALIDATION_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)


# ==================================================
# BASIC INFORMATION
# ==================================================

print()
print("Dataset sizes:")

print(
    "Train:",
    len(train_df)
)

print(
    "Validation:",
    len(validation_df)
)

print(
    "Test:",
    len(test_df)
)


# ==================================================
# CHECK COLUMNS
# ==================================================

print()
print("--------------------------------------")
print("Checking columns...")


expected_columns = [
    "text",
    "label",
    "source"
]


for name, df in [
    ("Train", train_df),
    ("Validation", validation_df),
    ("Test", test_df)
]:

    print(
        f"{name} columns:",
        list(df.columns)
    )

    for column in expected_columns:

        assert column in df.columns, (
            f"{column} missing from {name}"
        )


print("Column check: PASSED")


# ==================================================
# CHECK EMPTY TEXT
# ==================================================

print()
print("--------------------------------------")
print("Checking empty texts...")


for name, df in [
    ("Train", train_df),
    ("Validation", validation_df),
    ("Test", test_df)
]:

    empty_count = (
        df["text"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        f"{name} empty texts:",
        empty_count
    )

    assert empty_count == 0


print("Empty text check: PASSED")


# ==================================================
# CHECK LABELS
# ==================================================

print()
print("--------------------------------------")
print("Checking labels...")


for name, df in [
    ("Train", train_df),
    ("Validation", validation_df),
    ("Test", test_df)
]:

    labels = set(
        df["label"].unique()
    )

    print(
        f"{name} labels:",
        labels
    )

    assert labels.issubset(
        {0, 1}
    )


print("Label check: PASSED")


# ==================================================
# CHECK CLASS DISTRIBUTION
# ==================================================

print()
print("--------------------------------------")
print("Class distributions:")


for name, df in [
    ("Train", train_df),
    ("Validation", validation_df),
    ("Test", test_df)
]:

    print()
    print(name)

    print(
        df["label"].value_counts()
    )


# ==================================================
# CHECK DUPLICATES INSIDE EACH SPLIT
# ==================================================

print()
print("--------------------------------------")
print("Checking duplicates...")


for name, df in [
    ("Train", train_df),
    ("Validation", validation_df),
    ("Test", test_df)
]:

    duplicates = (
        df["text"].duplicated().sum()
    )

    print(
        f"{name} duplicate texts:",
        duplicates
    )

    assert duplicates == 0


# ==================================================
# CHECK DUPLICATES BETWEEN SPLITS
# ==================================================

print()
print("--------------------------------------")
print("Checking cross-split duplicates...")


train_texts = set(
    train_df["text"]
)

validation_texts = set(
    validation_df["text"]
)

test_texts = set(
    test_df["text"]
)


train_validation_overlap = (
    train_texts &
    validation_texts
)

train_test_overlap = (
    train_texts &
    test_texts
)

validation_test_overlap = (
    validation_texts &
    test_texts
)


print(
    "Train / Validation overlap:",
    len(train_validation_overlap)
)

print(
    "Train / Test overlap:",
    len(train_test_overlap)
)

print(
    "Validation / Test overlap:",
    len(validation_test_overlap)
)


assert len(train_validation_overlap) == 0

assert len(train_test_overlap) == 0

assert len(validation_test_overlap) == 0


print(
    "Cross-split duplicate check: PASSED"
)


# ==================================================
# FINAL SUMMARY
# ==================================================

print()
print("======================================")
print("DATASET VALIDATION COMPLETE")
print("======================================")

print()
print("All checks passed successfully.")