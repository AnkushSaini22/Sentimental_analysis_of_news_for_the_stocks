# Sentimental analysis of news for the stock


## Overview

This project implements a quantitative machine learning pipeline to predict stock market movements—specifically the Dow Jones Industrial Average (DJIA)—by combining historical market data with sentiment analysis of daily top news headlines.

The pipeline handles both directional classification (predicting whether the market will close UP or DOWN) and return regression (predicting the magnitude of the price change). It is designed with robust financial modeling principles, specifically addressing common pitfalls like data leakage and class imbalance.

## Dataset

The project utilizes a combined dataset consisting of:

* **Historical Stock Prices**: Daily Open, High, Low, Close, and Volume data for the DJIA.
* **Daily News Headlines**: The top 25 daily news headlines sourced from Reddit's WorldNews channel.

## Key Features & Feature Engineering

The model does not rely on raw prices, but rather constructs a feature set typical of quantitative trading strategies:

* **Sentiment Analysis (VADER)**: Extracts compound sentiment scores for individual headlines rather than aggregated text to prevent sentiment dilution. Captures the mean, minimum (most negative news), and maximum (most positive news) sentiment of the day.
* **Technical Indicators**: Utilizes the `ta` library to compute Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), and Bollinger Band width.
* **Price Action & Momentum**: Calculates 1-day and 3-day trailing returns.
* **Volatility Metrics**: Computes 7-day rolling standard deviation of returns.
* **Strict Temporal Alignment**: All price features (High, Low, Close, Volume) are shifted by 1 day to strictly prevent look-ahead bias and data leakage during training.

## Machine Learning Models

The pipeline explores multiple modeling approaches:

### 1. Directional Prediction (Classification)

Focuses on predicting the binary movement of the market (Up=1, Down=0).

* **Models Used**: Logistic Regression (L2 regularized) and Random Forest Classifier.
* **Ensemble Method**: Implements a Soft Voting Classifier combining Logistic Regression and a shallow Random Forest to reduce variance and prevent overfitting to market noise.
* **Class Imbalance**: Utilizes balanced class weights to ensure the model appropriately identifies negative market days.

### 2. Return Prediction (Regression)

Focuses on predicting the continuous return value.

* **Models Used**: Ridge Regression.
* **Regularization**: Applies strong L2 penalty (high alpha) to shrink coefficients, preventing the model from fitting to the extreme noise inherent in daily stock returns.

## Prerequisites

To run this notebook, you will need the following Python libraries installed:

```bash
pip install pandas numpy scikit-learn xgboost nltk vaderSentiment ta

```

You must also download the VADER lexicon within your Python environment:

```python
import nltk
nltk.download('vader_lexicon')

```

## Usage

1. Clone the repository and ensure the datasets (`Combined_News_DJIA(train).csv`, `DJIA_table(train).csv`, `Test_Combined_News.csv`, `Test_DJIA_Table.csv`) are located in the appropriate input directory.
2. Open the Jupyter Notebook `sentimental_analysis_of_news_for_the_stocks (1).ipynb`.
3. Run the cells sequentially. The notebook will:
* Clean the text data and extract VADER sentiment scores.
* Generate rolling technical and statistical features.
* Standardize the feature set.
* Train the classification and regression models.
* Output performance metrics (Accuracy, Precision, Recall, RMSE, R-squared).
* Generate a final `submission.csv` containing the predicted closing prices for the test set.



## Project Structure

* `sentimental_analysis_of_news_for_the_stocks (1).ipynb`: The main notebook containing the data processing, feature engineering, and modeling pipeline.
* `submission.csv`: The generated output file containing daily predicted closing prices.
