from image_dataset import ImageDataset


DATASET_PATH = "data/images"


dataset = ImageDataset(
    DATASET_PATH
)


print()
print("--------------------------------------")
print("IMAGE DATASET TEST")
print("--------------------------------------")

print(
    "Number of images:",
    len(dataset)
)


if len(dataset) > 0:

    sample = dataset[0]

    print()
    print("First image:")
    print(
        sample["image_path"]
    )

    print()
    print("Pixel tensor shape:")

    print(
        sample["pixel_values"].shape
    )