from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, Sequence


def build_vectorizer(train_texts: Sequence[str], ngram_range: Tuple[int, int] = (1, 2), max_features: int = 5000) -> Tuple[TfidfVectorizer, any]:
    """Create and fit a TF-IDF vectorizer on the training texts.

    Args:
        train_texts: Iterable of cleaned training text strings.
        ngram_range: Tuple specifying the range of n-grams to include.
        max_features: Maximum number of features to extract.

    Returns:
        A fitted TfidfVectorizer and the transformed training matrix.
    """
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=ngram_range, max_features=max_features)
    X = vectorizer.fit_transform(train_texts)
    return vectorizer, X
