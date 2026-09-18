import torch

from src.text_model import TextAIDetector


def test_model_output():

    model = TextAIDetector()

    input_ids = torch.randint(
        0,
        1000,
        (2, 256)
    )

    attention_mask = torch.ones(
        (2, 256),
        dtype=torch.long
    )

    output = model(
        input_ids,
        attention_mask
    )

    assert output.shape == (2, 2)