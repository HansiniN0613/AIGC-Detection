import pandas as pd
from metadata_features import extract_text_features


INPUT_FILE = "data/text/hc3_multisource_raw.csv"
OUTPUT_FILE = "data/text/hc3_with_text_metadata.csv"


print("=" * 60)
print("TEXT METADATA PREPARATION")
print("=" * 60)


df = pd.read_csv(INPUT_FILE)

print("Original samples:", len(df))


metadata_rows = []

for index, text in enumerate(df["text"]):

    if index % 1000 == 0:
        print(
            f"Processing {index}/{len(df)}"
        )

    features = extract_text_features(
        text
    )

    metadata_rows.append(features)


metadata_df = pd.DataFrame(
    metadata_rows
)


df = pd.concat(
    [
        df.reset_index(drop=True),
        metadata_df.reset_index(drop=True)
    ],
    axis=1
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("=" * 60)
print("COMPLETED")
print("=" * 60)

print("Total samples:", len(df))
print()
print("Columns:")
print(df.columns.tolist())

print()
print("Saved to:")
print(OUTPUT_FILE)