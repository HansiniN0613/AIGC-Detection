import pandas as pd

from transformers import RobertaTokenizer


# ==================================================
# SETTINGS
# ==================================================

DATASET_FILE = "data/text/train.csv"


# ==================================================
# LOAD DATASET
# ==================================================

print()
print("======================================")
print("TEXT LENGTH ANALYSIS")
print("======================================")


df = pd.read_csv(
    DATASET_FILE
)


# ==================================================
# LOAD TOKENIZER
# ==================================================

print()
print("Loading RoBERTa tokenizer...")


tokenizer = RobertaTokenizer.from_pretrained(
    "roberta-base"
)


# ==================================================
# CALCULATE TOKEN LENGTHS
# ==================================================

print()
print("Analyzing text lengths...")


token_lengths = []


for text in df["text"]:

    tokens = tokenizer(
        str(text),
        add_special_tokens=True,
        truncation=False
    )

    token_lengths.append(
        len(tokens["input_ids"])
    )


df["token_length"] = token_lengths


# ==================================================
# DISPLAY STATISTICS
# ==================================================

print()
print("--------------------------------------")

print(
    "Shortest text:",
    df["token_length"].min(),
    "tokens"
)

print(
    "Longest text:",
    df["token_length"].max(),
    "tokens"
)

print(
    "Average length:",
    round(
        df["token_length"].mean(),
        2
    ),
    "tokens"
)

print(
    "Median length:",
    df["token_length"].median(),
    "tokens"
)

print(
    "Texts <= 256 tokens:",
    (
        df["token_length"] <= 256
    ).sum()
)

print(
    "Texts <= 512 tokens:",
    (
        df["token_length"] <= 512
    ).sum()
)

print(
    "Texts > 512 tokens:",
    (
        df["token_length"] > 512
    ).sum()
)


# ==================================================
# PERCENTAGES
# ==================================================

total = len(df)


within_256 = (
    df["token_length"] <= 256
).sum()

within_512 = (
    df["token_length"] <= 512
).sum()


print()
print("--------------------------------------")

print(
    f"Within 256 tokens: "
    f"{within_256 / total * 100:.2f}%"
)

print(
    f"Within 512 tokens: "
    f"{within_512 / total * 100:.2f}%"
)

print(
    f"Over 512 tokens: "
    f"{(total - within_512) / total * 100:.2f}%"
)


print()
print("======================================")
print("ANALYSIS COMPLETE")
print("======================================")