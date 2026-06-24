# Twitter Airline Sentiment Analysis

This project provides a complete analysis pipeline for the Twitter US Airline Sentiment dataset and an interactive Streamlit application for sentiment prediction on airline tweets.

## Dataset

The Twitter US Airline Sentiment dataset contains 14 640 tweets posted in February 2015 about six major U.S. airlines (United Airlines, US Airways, American Airlines, Southwest Airlines, Delta Air Lines and Virgin America). Each tweet is labeled as positive, neutral or negative; negative tweets are further classified into specific reasons such as late flight, customer service or lost luggage. The dataset includes metadata like tweet ID, user information and geo‑coordinates. The data were crowd‑sourced by CrowdFlower and originally used for a sentiment analysis competition.

## Repository structure

- **src/** – Python modules for downloading the dataset, preprocessing text, vectorizing, training models and running predictions.
- **models/** – Serialized models and vectorizers (generated after running training scripts).
- **visuals/** – Plots generated during exploratory analysis.
- **app/** – Streamlit app that loads the trained model and allows interactive sentiment classification.
- **notebooks/** – (optional) Jupyter notebooks for additional exploration.
- **report.md** – A detailed report summarizing dataset exploration and modeling.
- **requirements.txt** – List of Python dependencies.
- **.gitignore** – Patterns to exclude from version control.

## Getting started

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download the dataset (if you don't already have it). The `src/download_data.py` script fetches the CSV from the public repository used in our analysis:

```bash
python src/download_data.py --output data/raw/airline_tweets.csv
```

3. Preprocess and train the model:

```bash
python src/train_model.py --data data/raw/airline_tweets.csv --model models/model.pkl --vectorizer models/vectorizer.pkl
```

4. Launch the Streamlit app:

```bash
streamlit run app/app.py
```

The app provides a simple interface where you can paste an airline-related tweet and receive the predicted sentiment.

## Citations

The dataset description and statistics included in this repository are derived from the CrowdFlower Twitter airline sentiment dataset.
