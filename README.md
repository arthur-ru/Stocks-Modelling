# Financial Data Analysis and Modeling Project

## Overview
This project involves collecting historical stock price data for S&P 500 companies, preprocessing the data by identifying outliers, and training a stock price prediction model using a GRU neural network. The code is divided into three main parts: data collection (`data_collection.py`), data preprocessing (`data_preprocessing.py`), and model training and evaluation (`model_training.py` and `model_evaluation.py`).

## Note
This project is currently in the development stage and is subject to change. Future updates may include improvements in data collection and processing methods, model accuracy, and the addition of new features.

## Installation
Ensure that the required libraries (e.g., `yfinance`, `pandas`, `numpy`, `torch`, `yaml`) are installed before running the scripts.  
Install PyTorch on your device by clicking [here](https://pytorch.org/), and getting the appropriate version for your distribution.
Install Cuda on your device by clicking [here](https://developer.nvidia.com/cuda-downloads) (we are at version 12.3 right now). This will allow the script to execute directly on the GPU to fasten the calculations.

## Data Collection (`data_collection.py`)
- `yfinance` library to download historical stock price data for S&P 500 companies from Yahoo Finance.
- S&P 500 company symbols are obtained from the Wikipedia page listing S&P 500 companies.
- Filtered data to include only valid symbols with available historical data for a specified period.
- Collected data is saved to a CSV file (`data/raw_data/data_stock.csv`), and symbols with no available data are saved to a text file (`data/raw_data/invalid_symbols.txt`).

## Data Preprocessing (`data_preprocessing.py`)
- Identifies outliers in the data using the Interquartile Range (IQR) method for each stock symbol.
- Outliers are saved to a separate CSV file (`data/raw_data/outliers.csv`).

## Model Training (`model_training.py`)
- Trains a GRU (Gated Recurrent Unit) neural network for stock price prediction.
- Parameters are specified in a YAML file (`config/config.yaml`).
- The model is trained on a specified stock symbol (e.g., "MSFT") using 80% of the data for training and 20% for testing.
- The trained model is saved to a file (`models/model_<symbol>.pth`).

## Model Evaluation (`model_evaluation.py`)
- Evaluates the trained model on the test set using the remaining 20% of the data.
- Calculates the average loss on the test set and prints the result.

## Configuration
- Model hyperparameters such as learning rate, batch size, hidden dimensions, and the number of GRU layers are specified in the configuration file (`config/config.yaml`).

## Usage
1. Run `data_collection.py` to collect and save stock price data.  
Change the `start_date` and `end_date` as you wish.
2. Run `data_preprocessing.py` to identify and save outliers in the data.
3. Run `model_training.py` to train a GRU model on a specific stock symbol.  
Change the `input_value` for the ticker you want to train your model on.
4. Run `model_evaluation.py` to evaluate the trained model on the test set.
5. Change the hyperparameters in `config.yaml` to reduce the execution time and accuracy of the model.

## Areas for improvement
Feel free to customize the project according to your specific needs and datasets!
Areas of improvement might be:
- Outliers management: defining if the outliers are true or not using another financial API (we haven't managed to find another free one allowing us for so much requests).
- Getting the sentiment of a financial news article (we haven't managed to find a free News API allowing us to see the content of financial articles) to determine the market trend.

## Contact
For questions or feedback, please contact me at [arthurrubio0@gmail.com](mailto:arthurrubio0@gmail.com).

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.