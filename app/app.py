import streamlit as st
import joblib
from pathlib import Path
import sys

# Add src to path for import
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from preprocess import clean_text  # noqa: E402

@st.cache_data
def load_model(model_path: str, vectorizer_path: str):
    """Load the trained model and vectorizer from disk."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def main():
    st.title("Twitter Airline Sentiment Classifier")
    st.write("Enter a tweet about a U.S. airline to predict the sentiment (negative/neutral/positive).")
    input_text = st.text_area("Tweet text", "")

    model_path = Path("models/model.pkl")
    vectorizer_path = Path("models/vectorizer.pkl")

    if not model_path.exists() or not vectorizer_path.exists():
        st.warning("Model not found. Please run the training script first to generate the model and vectorizer.")
        return

    if st.button("Predict") and input_text.strip():
        model, vectorizer = load_model(str(model_path), str(vectorizer_path))
        clean = clean_text(input_text)
        X = vectorizer.transform([clean])
        prediction = model.predict(X)[0]
        st.success(f"Predicted sentiment: {prediction.capitalize()}")


if __name__ == "__main__":
    main()
