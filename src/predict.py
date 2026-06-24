import argparse
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


def predict_sentiment(model_path: str, vectorizer_path: str, text: str) -> str:
    """Predict the sentiment label for a given tweet text using a trained model and vectorizer."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    clean = clean_text(text)
    X = vectorizer.transform([clean])
    return model.predict(X)[0]


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment for a tweet")
    parser.add_argument("--model", type=str, default="models/model.pkl", help="Path to trained model")
    parser.add_argument("--vectorizer", type=str, default="models/vectorizer.pkl", help="Path to vectorizer")
    parser.add_argument("--text", type=str, required=True, help="Tweet text to classify")
    args = parser.parse_args()
    sentiment = predict_sentiment(args.model, args.vectorizer, args.text)
    print(f"Predicted sentiment: {sentiment}")


if __name__ == "__main__":
    main()
