# US Airline Twitter Sentiment Analysis

End-to-end NLP project that classifies U.S. airline tweets into sentiment categories.

## Business Problem

Airlines receive thousands of public messages on social media. Understanding customer sentiment helps carriers identify pain points, improve service, and respond proactively. Manual review of tweets is time-consuming and subjective. This project builds a machine-learning system that automatically predicts the sentiment of tweets directed at major U.S. airlines as **positive**, **neutral**, or **negative**.

The goal is to provide airlines with timely customer-experience insights that can support faster response, better service recovery, and improved operational decision-making.

## Dataset

**Source:** CrowdFlower / Twitter US Airline Sentiment dataset  
**Project type:** Natural Language Processing / Sentiment Classification  
**Domain:** Customer Experience Analytics

The dataset contains **14,640 tweets** posted in February 2015 about six U.S. airlines:

- United Airlines
- US Airways
- American Airlines
- Southwest Airlines
- Delta Air Lines
- Virgin America

Each tweet is labeled as **positive**, **neutral**, or **negative**. Negative tweets also include specific negative-reason categories such as late flight, rude service, canceled flight, customer service issues, lost luggage, and booking problems.

The dataset includes tweet text, airline name, sentiment label, sentiment confidence score, negative reason, retweet count, location fields, and timestamp-related metadata.

## Project Objectives

- Build a reproducible NLP pipeline for airline tweet sentiment classification.
- Clean and transform raw social media text into machine-learning-ready features.
- Train a baseline sentiment classifier using TF-IDF and Logistic Regression.
- Evaluate model performance using standard classification metrics.
- Identify common themes in negative airline feedback.
- Create a Streamlit app that allows users to enter a tweet and receive a sentiment prediction.
- Package the project as a resume-ready, interview-ready GitHub portfolio project.

## Methods

- **Data ingestion:** Load the airline sentiment dataset from a reproducible source.
- **Text cleaning:** Remove URLs, mentions, punctuation, numbers, and extra whitespace; convert text to lowercase.
- **Feature engineering:** Convert cleaned tweets into numerical features using TF-IDF vectorization with unigram and bigram tokens.
- **Modeling:** Train a Logistic Regression classifier as a strong NLP baseline.
- **Evaluation:** Measure accuracy, precision, recall, F1-score, and confusion matrix results.
- **Business analysis:** Review sentiment distribution by airline and examine common negative feedback reasons.
- **Deployment-ready app:** Build a Streamlit application for interactive sentiment prediction.

## Repository Structure

```text
Twitter_Airline_Sentiment/
├── app/
│   └── app.py                   # Streamlit app for interactive prediction
├── data/
│   └── raw/                     # Directory for downloaded dataset; raw data not committed
├── models/                      # Serialized model and vectorizer saved after training
├── notebooks/                   # Optional notebooks for exploration and documentation
├── src/
│   ├── download_data.py         # Script to download the dataset
│   ├── preprocess.py            # Text cleaning utilities
│   ├── feature_engineering.py   # TF-IDF vectorizer builder
│   ├── train_model.py           # Model training script
│   └── predict.py               # Command-line prediction utility
├── visuals/                     # Generated charts and model visuals
├── report.md                    # Detailed project analysis report
├── requirements.txt             # Python dependencies
├── .gitignore                   # Files and folders excluded from Git
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Rachel-Oyeyemi/Twitter_Airline_Sentiment.git
cd Twitter_Airline_Sentiment
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

```bash
python src/download_data.py --output data/raw/airline_tweets.csv
```

### 5. Train the model

```bash
python src/train_model.py --data data/raw/airline_tweets.csv --model models/model.pkl --vectorizer models/vectorizer.pkl
```

### 6. Run the Streamlit app

```bash
streamlit run app/app.py
```

## Example App Use Case

**Input tweet:**

```text
Just landed after a 4 hour delay and the gate agent was rude. Never flying with this airline again!
```

**Expected prediction:**

```text
Negative
```

## Model Performance

The baseline model uses **TF-IDF + Logistic Regression**, a practical and interpretable NLP approach for text classification. The model achieved approximately **78% accuracy** on the test set.

Performance was strongest for the negative class because negative tweets make up the largest share of the dataset. Neutral and positive classes were more challenging, which is common in sentiment analysis when short social media text contains ambiguous language.

## Business Insights

- Negative tweets dominate the dataset, suggesting that customers are more likely to post airline feedback when they experience service problems.
- Common negative reasons include delayed flights, customer service issues, cancellations, and lost luggage.
- Sentiment patterns can help airline support teams prioritize urgent service failures and identify recurring customer pain points.
- A sentiment monitoring tool can support social media triage, brand reputation tracking, and customer experience improvement.

## Key Deliverables

- Reproducible Python machine-learning pipeline
- Text preprocessing and TF-IDF feature engineering scripts
- Logistic Regression sentiment classification model
- Model evaluation report with classification metrics
- Streamlit web app for live tweet sentiment prediction
- Resume-ready README documentation
- Business-focused insights and recommendations

## Resume Bullets

- Built an NLP classification system using TF-IDF and Logistic Regression to categorize U.S. airline tweets as positive, neutral, or negative.
- Developed a reproducible machine-learning pipeline covering data ingestion, text cleaning, feature extraction, model training, evaluation, and app deployment.
- Created an interactive Streamlit application that predicts tweet sentiment and supports real-time customer-experience analysis.
- Analyzed airline sentiment distribution and negative-reason categories to identify recurring operational pain points such as delays, cancellations, and service issues.

## LinkedIn Project Description

Completed an end-to-end NLP portfolio project analyzing U.S. airline customer sentiment from Twitter data. The project includes text preprocessing, TF-IDF feature engineering, Logistic Regression classification, model evaluation, business insight generation, and an interactive Streamlit app for real-time sentiment prediction. This project demonstrates applied machine learning, NLP, customer experience analytics, and deployment-ready portfolio documentation.

## Future Improvements

- Fine-tune transformer models such as BERT or RoBERTa to improve classification performance.
- Address class imbalance using class weights, oversampling, or additional training data.
- Incorporate airline, timestamp, and location metadata as additional model features.
- Add SHAP or LIME explanations to improve model interpretability.
- Deploy the Streamlit app to Streamlit Community Cloud.
- Build a dashboard that tracks sentiment trends by airline and negative-reason category.

## About

This project was created as part of a professional Data Analytics and AI portfolio. It is designed to demonstrate skills in Python, NLP, machine learning, text preprocessing, model evaluation, Streamlit app development, and business-focused data storytelling.
