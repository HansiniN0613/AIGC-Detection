import json
import os
import pandas as pd
from huggingface_hub import hf_hub_download


# ======================================
# SETTINGS
# ======================================

REPO_ID = "Hello-SimpleAI/HC3"

SOURCE_FILES = [
    "reddit_eli5.jsonl",
    "open_qa.jsonl",
    "wiki_csai.jsonl",
    "medicine.jsonl",
    "finance.jsonl"
]

OUTPUT_FILE = "data/text/hc3_multisource_raw.csv"

MAX_RECORDS_PER_SOURCE = 1000


# ======================================
# START
# ======================================

print()
print("======================================")
print("HC3 MULTI-SOURCE DATASET PREPARATION")
print("======================================")
print()


all_data = []


# ======================================
# PROCESS EACH SOURCE
# ======================================

for source_file in SOURCE_FILES:

    print("--------------------------------------")
    print("Processing:", source_file)
    print("--------------------------------------")

    try:

        file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=source_file,
            repo_type="dataset"
        )

        print("Downloaded successfully.")

    except Exception as error:

        print("Could not download:", source_file)
        print("Error:", error)
        continue


    records = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(file):

            if line_number >= MAX_RECORDS_PER_SOURCE:
                break

            record = json.loads(line)

            records.append(record)


    print(
        "Questions loaded:",
        len(records)
    )


    # ==================================
    # CREATE SAMPLES
    # ==================================

    source_name = source_file.replace(
        ".jsonl",
        ""
    )

    for record in records:

        # ------------------------------
        # HUMAN ANSWERS
        # ------------------------------

        human_answers = record.get(
            "human_answers",
            []
        )

        for answer in human_answers:

            if answer and answer.strip():

                all_data.append({
                    "text": answer.strip(),
                    "label": 0,
                    "source": source_name
                })


        # ------------------------------
        # AI ANSWERS
        # ------------------------------

        chatgpt_answers = record.get(
            "chatgpt_answers",
            []
        )

        for answer in chatgpt_answers:

            if answer and answer.strip():

                all_data.append({
                    "text": answer.strip(),
                    "label": 1,
                    "source": source_name
                })


# ======================================
# CREATE DATAFRAME
# ======================================

print()
print("Creating dataframe...")

df = pd.DataFrame(all_data)


# ======================================
# REMOVE DUPLICATES
# ======================================

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


df = df.reset_index(drop=True)


# ======================================
# SAVE
# ======================================

os.makedirs(
    "data/text",
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ======================================
# SUMMARY
# ======================================

print()
print("======================================")
print("DATASET PREPARATION COMPLETE")
print("======================================")

print()
print("Total samples:", len(df))

print()
print("Label distribution:")

print(
    df["label"].value_counts()
)

print()
print("Source distribution:")

print(
    df.groupby(
        ["source", "label"]
    ).size()
)

print()
print("Saved to:")
print(OUTPUT_FILE)