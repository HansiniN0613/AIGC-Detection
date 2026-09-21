import torch

from PIL import Image

from transformers import ViTImageProcessor

from image_model import ImageAIDetector


# ---------------------------------------------------
# SETTINGS

MODEL_PATH = "models/image_detector.pth"


# ---------------------------------------------------
# DEVICE

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------------------
# IMAGE PROCESSOR

processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)


# -------------------------------------------------
# LOAD MODEL

model = ImageAIDetector()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# ---------------------------------------------------
# PREDICTION FUNCTION

def predict_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    encoding = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = encoding[
        "pixel_values"
    ].to(device)

    with torch.no_grad():

        logits = model(
            pixel_values
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    human_probability = (
        probabilities[0][0].item()
    )

    ai_probability = (
        probabilities[0][1].item()
    )

    ai_percentage = (
        ai_probability * 100
    )

    return {
        "human_probability":
            human_probability,

        "ai_probability":
            ai_probability,

        "ai_percentage":
            ai_percentage
    }


# -------------------------------------------------
# RUN PROGRAM

if __name__ == "__main__":

    print()
    print("**************************************")
    print("AI-GENERATED IMAGE DETECTOR")
    print("**************************************")

    image_path = input(
        "\nEnter image path: "
    )

    result = predict_image(
        image_path
    )

    print()
    print("--------------------------------------")

    print(
        f"AI-generated probability: "
        f"{result['ai_percentage']:.2f}%"
    )

    print(
        f"Human probability: "
        f"{result['human_probability'] * 100:.2f}%"
    )

    print("--------------------------------------")