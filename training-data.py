import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

# Load the data source into a returned dataframe variable
def load_training_data(file_path, num, cat):
    # Read data source into pandas dataframe (df)
    cols = num + cat
    df = pd.read_csv(file_path, nrows=2000, usecols=cols)
    df["earliest_cr_line"] = pd.to_datetime(df['earliest_cr_line'], format='%b-%y')
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%y")
    df["last_pymnt_d"] = pd.to_datetime(df["last_pymnt_d"], format="%b-%y")
    # df["payment_length"] = df["last_pymnt_d"] - df["issue_d"]
    df["payment_length_months"] = ((df["last_pymnt_d"].dt.year - df["issue_d"].dt.year) * 12 +
                                   (df["last_pymnt_d"].dt.month - df["issue_d"].dt.month))
    # Print head of df
    # 5 is default
    print(df.head(25))
    return df


def process_data(df, numeric, categorical):
    X = df[numeric + categorical]

    # Generate the mean 
    preprocess = ColumnTransformer([
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ])

    Xt = preprocess.fit_transform(X)
    print(Xt[0:5])
    return Xt

# Pre-processing the data to remove anomalies, missing values, etc.
# Drop rows where data values aren't mapped correctly eg: NaN, 
# def clean_data(df):


# Train a machine learning model using the global dataframe and return its output
def train_model(Xt, df, numeric):
    pass

if __name__ == "__main__":
    # Configure pandas
    pd.set_option('display.max_columns', None)
    # Specify data source
    file_path = "loan_data.csv"
    numeric = [
    "loan_amnt", "int_rate",
    "annual_inc", "delinq_2yrs", "open_acc", "pub_rec",
    "revol_bal", "repay_fail", "earliest_cr_line"
    ]
    categorical = [
    "term", "emp_length", "home_ownership", "issue_d",
    "purpose",
    "last_pymnt_d"
    ]
    # Load training data into model
    data = load_training_data(file_path, numeric, categorical)
    # preprocess = process_data(data, numeric, categorical)
    # output = train_model(preprocess, data, numeric)