from metadata_features import extract_metadata


text = """
Check out this amazing place! 
What do you think? #travel #srilanka
"""


image_path = "data/images/moon_stone.png"


features = extract_metadata(
    text,
    image_path
)


print()
print("=" * 50)
print("METADATA FEATURES")
print("=" * 50)

for feature, value in features.items():

    print(
        f"{feature}: {value}"
    )