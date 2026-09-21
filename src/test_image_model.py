import torch

from image_model import ImageAIDetector


print()
print("--------------------------------------")
print("IMAGE MODEL TEST")
print("--------------------------------------")


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "Using device:",
    device
)


print()
print("Loading ViT model...")


model = ImageAIDetector()

model = model.to(device)


print()
print("Creating test image tensor...")


pixel_values = torch.randn(
    1,
    3,
    224,
    224
)


pixel_values = pixel_values.to(device)


print(
    "Input shape:",
    pixel_values.shape
)


print()
print("Running model...")


with torch.no_grad():

    output = model(
        pixel_values
    )


print()
print(
    "Output shape:",
    output.shape
)


print()
print("Output:")

print(output)


print()
print("--------------------------------------")
print("IMAGE MODEL TEST COMPLETE")
print("--------------------------------------")