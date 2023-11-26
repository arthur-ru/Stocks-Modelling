import pandas as pd
import numpy as np

# Load data from the "data_stock.csv" CSV file
data = pd.read_csv("data/raw_data/data_stock.csv")

# Get the column names (except the date column)
column_names = data.columns[1:]  # Start from the second column

# Create a dictionary to store the outliers
outliers_data = {}

# Loop over each column and identify outliers using the IQR method
for column_name in column_names:
    ticker_data = data[column_name]
    Q1 = np.percentile(ticker_data, 25)
    Q3 = np.percentile(ticker_data, 75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    outliers_IQR = (ticker_data < lower_limit) | (ticker_data > upper_limit)

    # Add outliers for this column to the dictionary
    outliers_data[column_name] = ticker_data[outliers_IQR]

    # Print the outliers for this column
    print(f"Outliers identified with the IQR method for {column_name}:")
    print(ticker_data[outliers_IQR])

# Create a DataFrame from the outliers dictionary
outliers_df = pd.DataFrame(outliers_data)

# Add the other columns from the original DataFrame (including the date column)
outliers_df = pd.concat([data.iloc[:, 0], outliers_df], axis=1)

# Save the outliers to a CSV file
outliers_df.to_csv("data/raw_data/outliers.csv", index=False)