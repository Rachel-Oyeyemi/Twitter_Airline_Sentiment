# Twitter US Airline Sentiment Analysis Report

This document summarises an exploratory analysis and modelling of the Twitter US Airline Sentiment dataset.

## Overview

The Twitter US Airline Sentiment dataset contains 14 640 tweets posted in February 2015 about six major U.S. airlines (United Airlines, US Airways, American Airlines, Southwest Airlines, Delta Air Lines and Virgin America). Each tweet is labelled as positive, neutral or negative. Negative tweets are further annotated with specific reasons such as "late flight", "rude service", "canceled flight" or "lost luggage". The dataset also includes metadata such as tweet ID, user information and coordinates, and is small (≈8.5 MB) yet rich for sentiment‑analysis experiments.

## Data exploration

- **Sentiment distribution** – Negative tweets account for the majority of the dataset, with neutral tweets roughly half as frequent and positive tweets being the smallest class. This imbalance has implications for model performance.
- **Negative reasons** – Customer service issues, late flights and cancelled flights are among the most common reasons for negative sentiment. Other categories include lost luggage and problems related to ticket purchases or amenities.
- **Sentiment by airline** – Virgin America receives a comparatively higher share of positive tweets, while United Airlines and American Airlines attract more negative comments. Southwest and Delta exhibit more balanced distributions with larger neutral proportions.

## Modelling approach

Tweets were cleaned by removing URLs, mentions, punctuation and converting to lowercase. A term‑frequency/inverse‑document‑frequency (TF‑IDF) vectoriser using unigrams and bigrams with English stop‑words removed transformed the text into features. A logistic regression classifier with L2 regularisation was trained on 80 % of the data; performance was evaluated on the remaining 20 %.

### Results

The model achieved an overall accuracy of approximately **78 %**. Precision, recall and F1‑scores were highest for negative tweets. Neutral and positive tweets were harder to classify, with neutral tweets showing the lowest recall. The confusion matrix indicated that many positive tweets were misclassified as neutral or negative, and neutral tweets were often labelled negative.

## Conclusions and future work

The analysis highlights the dominance of negative sentiment in airline‑related tweets and identifies common pain points for travellers. Although a simple TF‑IDF + logistic regression model performs reasonably well overall, there is room for improvement, especially in recognising neutral and positive tweets. Potential enhancements include:

- **Addressing class imbalance** via resampling or class‑weighted losses.
- **Using contextual language models** (e.g., BERT or RoBERTa) to better capture semantics.
- **Incorporating metadata** such as airline, timestamp or user location as additional features.

This repository provides scripts to download the data, preprocess tweets, train a sentiment classifier and deploy an interactive Streamlit app for real‑time prediction.
