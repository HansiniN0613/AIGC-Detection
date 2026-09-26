import os
import re
import string

from PIL import Image


def count_emojis(text):
    """
    Simple emoji counter.
    """

    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\u2600-\u26FF"
        "\u2700-\u27BF"
        "]",
        flags=re.UNICODE
    )

    return len(emoji_pattern.findall(text))


def extract_text_features(text):

    text = str(text)

    characters = len(text)

    words = text.split()

    word_count = len(words)

    if word_count > 0:
        avg_word_length = sum(
            len(word)
            for word in words
        ) / word_count
    else:
        avg_word_length = 0.0

    sentence_count = len(
        re.findall(
            r"[.!?]+",
            text
        )
    )

    hashtag_count = len(
        re.findall(
            r"#\w+",
            text
        )
    )

    mention_count = len(
        re.findall(
            r"@\w+",
            text
        )
    )

    url_count = len(
        re.findall(
            r"https?://\S+|www\.\S+",
            text
        )
    )

    emoji_count = count_emojis(text)

    letters = [
        char
        for char in text
        if char.isalpha()
    ]

    if len(letters) > 0:

        uppercase_ratio = sum(
            char.isupper()
            for char in letters
        ) / len(letters)

    else:

        uppercase_ratio = 0.0

    if characters > 0:

        punctuation_count = sum(
            char in string.punctuation
            for char in text
        )

        punctuation_ratio = (
            punctuation_count / characters
        )

    else:

        punctuation_ratio = 0.0

    return {
        "text_length": characters,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length,
        "hashtag_count": hashtag_count,
        "mention_count": mention_count,
        "url_count": url_count,
        "emoji_count": emoji_count,
        "uppercase_ratio": uppercase_ratio,
        "punctuation_ratio": punctuation_ratio
    }


def extract_image_features(image_path):

    if not image_path or not os.path.exists(image_path):

        return {
            "image_count": 0,
            "image_width": 0,
            "image_height": 0,
            "image_aspect_ratio": 0.0,
            "image_file_size": 0,
            "image_format": "NONE"
        }

    try:

        with Image.open(image_path) as image:

            width, height = image.size

            if height > 0:
                aspect_ratio = width / height
            else:
                aspect_ratio = 0.0

            image_format = image.format or "UNKNOWN"

        file_size = os.path.getsize(
            image_path
        )

        return {
            "image_count": 1,
            "image_width": width,
            "image_height": height,
            "image_aspect_ratio": aspect_ratio,
            "image_file_size": file_size,
            "image_format": image_format
        }

    except Exception as e:

        print(
            f"Could not read image: {image_path}"
        )

        print("Error:", e)

        return {
            "image_count": 0,
            "image_width": 0,
            "image_height": 0,
            "image_aspect_ratio": 0.0,
            "image_file_size": 0,
            "image_format": "UNKNOWN"
        }


def extract_metadata(text, image_path):

    text_features = extract_text_features(
        text
    )

    image_features = extract_image_features(
        image_path
    )

    return {
        **text_features,
        **image_features
    }