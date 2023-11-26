# Financial Data Analysis and Modeling Project

## Overview
This project is a work in progress, focusing on the collection, preprocessing, and analysis of financial data, particularly stock prices. It involves using various Python scripts and Jupyter Notebooks to handle different aspects of data handling and machine learning modeling.

### Project Structure
- `data/`: Directory for storing data
    - `raw_data/`: Raw data downloaded from the API
    - `processed_data/`: Processed data ready for modeling

- `notebooks/`: Directory for Jupyter Notebooks
    - `01_data_collection.ipynb`: Notebook for data collection from Yahoo Finance API
    - `02_data_preprocessing.ipynb`: Notebook for data preprocessing
    - `03_model_training.ipynb`: Notebook for model training
    - `04_model_evaluation.ipynb`: Notebook for model evaluation

- `scripts/`: Directory for Python scripts
    - `data_collection.py`: Script for collecting data from the API
    - `data_preprocessing.py`: Script for data preprocessing
    - `model_training.py`: Script for model training
    - `model_evaluation.py`: Script for model evaluation

- `models/`: Directory for storing trained models

- `config/`: Directory for configuration files
    - `config.yaml`: Configuration file for model hyperparameters

- `test/`: Directory for code tests
    - `news_api.py`: Sentiment analyzing of financial articles

- `README.md`: Project documentation


## Data Collection (`data_collection.py`)
This script automates the collection of financial data, focusing on stock symbols from the S&P 500 list. The historical data for a specified period is fetched and stored, with the ability to filter out valid and invalid symbols.

## Data Preprocessing (`data_preprocessing.py`)
The preprocessing script is responsible for preparing the collected data for analysis. It involves cleaning, normalizing, and detecting outliers using the IQR method.

## Model Training (`model_training.py`)
This script is designed for training the financial data model. It includes steps for loading processed data, setting up various machine learning models, adjusting hyperparameters, and running the training process.

## Model Evaluation (`model_evaluation.py`)
This file includes the implementation of a GRU (Gated Recurrent Unit) network for the evaluation of stock price data. It uses PyTorch for building and training the machine learning model.

## Sentiment Analysis of Financial Articles (`news_api.py`)
This script is designed for sentiment analysis of financial news articles. It leverages APIs to fetch the latest news related to finance and stock markets, and applies natural language processing (NLP) techniques to analyze the sentiment (positive, negative, neutral) of the text. This analysis helps in understanding the impact of current news on stock market trends and investor sentiment.

## Note
This project is currently in the development stage and is subject to change. Future updates may include improvements in data collection and processing methods, model accuracy, and the addition of new features.