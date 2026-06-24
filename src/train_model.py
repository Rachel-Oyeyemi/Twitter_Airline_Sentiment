import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Local imports
def _add_src_to_path():
    import sys
    from pathlib import Path
    src_path = Path(__file__).resolve().parent
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))
_add_src_to_path()

from preprocess import clean_text
from feature_engineering import build_vectorizer


def train_model(data_path: str, model_path: str, vectorizer_path: str, test_size: float = 0.2, max_features: int = 5000) -> None:
    """Train a logistic regression classifier on the Twitter airline dataset and save the model and vectorizer."""
    df = pd.read_csv(data_path)
    df['clean_text'] = df['text'].astype(str).apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'],
        df['airline_sentiment'],
        test_size=test_size,
        stratify=df['airline_sentiment'],
        random_state=42
    )

    vectorizer, X_train_vec = build_vectorizer(X_train, max_features=max_features)

    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(X_train_vec, y_train)

    # Evaluation
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred))

    # Persist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")


def main():
    parser = argparse.ArgumentParser(description="Train sentiment classifier on the Twitter airline dataset")
    parser.add_argument("--data", type=str, default="data/raw/airline_tweets.csv", help="Path to CSV dataset")
    parser.add_argument("--model", type=str, default="models/model.pkl", help="Path to save model")
    parser.add_argument("--vectorizer", type=str, default="models/vectorizer.pkl", help="Path to save vectorizer")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    args = parser.parse_args()
    train_model(args.data, args.model, args.vectorizer, args.test_size)


if __name__ == "__main__":
    main()
