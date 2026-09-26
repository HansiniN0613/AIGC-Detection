import pandas as pd
from metadata_features import extract_text_features


INPUT_FILES = {
    "train": "data/text/train.csv",
    "validation": "data/text/validation.csv",
    "test": "data/text/test.csv"
}

OUTPUT_FILES = {
    "train": "data/text/train_metadata.csv",
    "validation": "data/text/validation_metadata.csv",
    "test": "data/text/test_metadata.csv"
}


print("=" * 60)
print("SPLIT METADATA PREPARATION")
print("=" * 60)


for split_name in ["train", "validation", "test"]:

    input_file = INPUT_FILES[split_name]
    output_file = OUTPUT_FILES[split_name]

    print()
    print("-" * 60)
    print(f"Processing: {split_name.upper()}")
    print("-" * 60)

    df = pd.read_csv(input_file)

    print("Samples:", len(df))

    metadata_rows = []

    for index, text in enumerate(df["text"]):

        if index % 1000 == 0:
            print(
                f"Processing {index}/{len(df)}"
            )

        features = extract_text_features(text)

        metadata_rows.append(features)

    metadata_df = pd.DataFrame(metadata_rows)

    df = pd.concat(
        [
            df.reset_index(drop=True),
            metadata_df.reset_index(drop=True)
        ],
        axis=1
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(f"Saved: {output_file}")


print()
print("=" * 60)
print("ALL SPLITS COMPLETED")
print("=" * 60)