import json
import os

import pandas as pd

from huggingface_hub import hf_hub_download


# -------------------------------------------------
# SETTINGS

REPO_ID = "Hello-SimpleAI/HC3"

FILE_NAME = "all.jsonl"

OUTPUT_FILE = "data/text/hc3_raw.csv"

MAX_QUESTIONS = 2000


# -------------------------------------------------
# START

print()
print("**************************************")
print("HC3 TEXT DATASET PREPARATION")
print("**************************************")


# -------------------------------------------------
# DOWNLOAD DATASET FILE

print()
print("Downloading HC3 JSONL file...")


file_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILE_NAME,
    repo_type="dataset"
)


print()
print("HC3 file downloaded.")

print(
    "File location:",
    file_path
)


# -------------------------------------------------
# READ JSONL

print()
print("Reading HC3 data...")


records = []


with open(
    file_path,
    "r",
    encoding="utf-8"
) as file:

    for line_number, line in enumerate(file):

        if line_number >= MAX_QUESTIONS:
            break

        record = json.loads(line)

        records.append(record)


print()
print(
    "Questions loaded:",
    len(records)
)


# -------------------------------------------------
# CONVERT TO TEXT + LABEL

print()
print("Creating text dataset...")


data = []


for record in records:

    # ----------------------------------------------
    # Human answers
    # ----------------------------------------------

    human_answers = record.get(
        "human_answers",
        []
    )

    for answer in human_answers:

        if answer and answer.strip():

            data.append({
                "text": answer.strip(),
                "label": 0,
                "source": record.get(
                    "source",
                    "unknown"
                )
            })


    # ----------------------------------------------
    # ChatGPT answers
    # ----------------------------------------------

    chatgpt_answers = record.get(
        "chatgpt_answers",
        []
    )

    for answer in chatgpt_answers:

        if answer and answer.strip():

            data.append({
                "text": answer.strip(),
                "label": 1,
                "source": record.get(
                    "source",
                    "unknown"
                )
            })


# -------------------------------------------------
# CREATE DATAFRAME

df = pd.DataFrame(data)


# -------------------------------------------------
# REMOVE DUPLICATES

print()
print("Removing duplicate texts...")


before = len(df)

df = df.drop_duplicates(
    subset=["text"]
)

after = len(df)


print(
    "Duplicates removed:",
    before - after
)


# -------------------------------------------------
# RESET INDEX

df = df.reset_index(
    drop=True
)


# -------------------------------------------------
# SAVE

os.makedirs(
    "data/text",
    exist_ok=True
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


# -------------------------------------------------
# SUMMARY

print()
print("**************************************")
print("DATASET PREPARATION COMPLETE")
print("**************************************")

print()
print(
    "Total text samples:",
    len(df)
)

print()
print("Label distribution:")

print(
    df["label"].value_counts()
)

print()
print("Label meaning:")

print("0 = Human")
print("1 = AI-generated")

print()
print(
    "Saved to:",
    OUTPUT_FILE
)