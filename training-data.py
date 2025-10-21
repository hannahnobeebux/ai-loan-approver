import pandas as pd

def load_training_data(file_path):
    # Read data source into pandas dataframe (df)
    df = pd.read_csv(file_path)
    # Print head of df
    print(df.head())
    return df


if __name__ == "__main__":
    # Configure pandas
    pd.set_option('display.max_columns', None)
    # Specify data source
    file_path = "credit_risk_dataset.csv"
    # Load training data into model
    data = load_training_data(file_path)