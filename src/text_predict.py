import torch

from transformers import RobertaTokenizer

from text_model import TextAIDetector


# --------------------------------------------------
# SETTINGS

MODEL_PATH = "models/text_detector.pth"


# --------------------------------------------------
# DEVICE

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# TOKENIZER

tokenizer = RobertaTokenizer.from_pretrained(
    "roberta-base"
)


# --------------------------------------------------
# LOAD MODEL

model = TextAIDetector()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# --------------------------------------------------
# PREDICTION FUNCTION

def predict_text(text):

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
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

    ai_percentage = (
        ai_probability * 100
    )

    return {
        "human_probability": human_probability,
        "ai_probability": ai_probability,
        "ai_percentage": ai_percentage
    }


# --------------------------------------------------
# RUN PROGRAM

if __name__ == "__main__":

    print()
    print("---------------------------------------")
    print("AI-GENERATED TEXT DETECTOR")
    print("--------------------------------------")

    text = input(
        "\nEnter text to analyze: "
    )

    result = predict_text(text)

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