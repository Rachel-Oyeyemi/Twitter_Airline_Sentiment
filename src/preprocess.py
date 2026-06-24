import re


def clean_text(text: str) -> str:
    """Clean tweet text by removing URLs, mentions, non-letter characters and extra spaces."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)  # remove mentions
    text = re.sub(r"[^a-z\s]", "", text)  # remove non-letters
    text = re.sub(r"\s+", " ", text).strip()
    return text
