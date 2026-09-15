import torch

from transformers import RobertaTokenizer

from text_model import TextAIDetector


MODEL_PATH = "../models/text_detector.pth"


# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Tokenizer

tokenizer = RobertaTokenizer.from_pretrained(
    "roberta-base"
)


# Load model

model = TextAIDetector()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# Prediction function

def predict_text(text):

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    input_ids = encoding[
        "input_ids"
    ].to(device)

    attention_mask = encoding[
        "attention_mask"
    ].to(device)

    with torch.no_grad():

        logits = model(
            input_ids,
            attention_mask
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

    return {
        "human_probability": human_probability,
        "ai_probability": ai_probability,
        "ai_percentage": ai_probability * 100
    }


# Test

if __name__ == "__main__":

    text = input(
        "\nEnter text to analyze: "
    )

    result = predict_text(text)

    print("\nResult")
    print("--------------------")

    print(
        f"Human probability: "
        f"{result['human_probability'] * 100:.2f}%"
    )

    print(
        f"AI probability: "
        f"{result['ai_percentage']:.2f}%"
    )