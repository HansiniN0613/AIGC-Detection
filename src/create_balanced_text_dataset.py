import pandas as pd


# -------------------------------------------------
# SETTINGS

INPUT_FILE = "data/text/hc3_raw.csv"

OUTPUT_FILE = "data/text/hc3_balanced.csv"

SAMPLES_PER_CLASS = 2000

RANDOM_STATE = 42


# -------------------------------------------------
# LOAD DATASET

print()
print("**************************************")
print("CREATING BALANCED TEXT DATASET")
print("**************************************")


df = pd.read_csv(
    INPUT_FILE
)


print()
print("Original dataset size:")
print(len(df))


print()
print("Original label distribution:")
print(df["label"].value_counts())


# -------------------------------------------------
# SEPARATE CLASSES

human_data = df[
    df["label"] == 0
]

ai_data = df[
    df["label"] == 1
]


# -------------------------------------------------
# SAMPLE DATA

human_sample = human_data.sample(
    n=SAMPLES_PER_CLASS,
    random_state=RANDOM_STATE
)

ai_sample = ai_data.sample(
    n=SAMPLES_PER_CLASS,
    random_state=RANDOM_STATE
)


# -------------------------------------------------
# COMBINE

balanced_df = pd.concat(
    [
        human_sample,
        ai_sample
    ],
    ignore_index=True
)


# -------------------------------------------------
# SHUFFLE

balanced_df = balanced_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(
    drop=True
)


# -------------------------------------------------
# SAVE

balanced_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# -------------------------------------------------
# SUMMARY

print()
print("**************************************")
print("BALANCED DATASET CREATED")
print("**************************************")


print()
print("Total samples:")
print(len(balanced_df))


print()
print("Label distribution:")
print(
    balanced_df["label"].value_counts()
)


print()
print("Saved to:")
print(OUTPUT_FILE)