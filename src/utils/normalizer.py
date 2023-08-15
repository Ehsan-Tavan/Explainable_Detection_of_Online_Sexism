import re


def normalize_text(input_data: str):
    """

    Args:
        input_data:

    Returns:

    """
    input_data = input_data.replace("[USER]", "")
    input_data = input_data.replace("[URL]", "")

    input_data = input_data.lower()

    # remove numbers
    input_data = re.sub(r"\d+", "", input_data)

    # remove all punctuation except words and space
    # input_data = re.sub(r"[^\w\s]", "", input_data)

    # remove white spaces
    input_data = input_data.strip()

    return input_data


if __name__ == "__main__":
    a = "[USER] Yea, animals are nice, I've been sick and my Tabby loves on me more, he knows, " \
        "they are smart. I got a bad habit of spoiling animals though😁"
