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
    # Calculating payment length in months
    df["payment_length_months"] = ((df["last_pymnt_d"].dt.year - df["issue_d"].dt.year) * 12 +
                                   (df["last_pymnt_d"].dt.month - df["issue_d"].dt.month))
    # Print head of df
    # 5 is default
    # print(df.head(25))
    return df


# Purpose of function: Cleaning the data to remove anomalies, missing values, etc.
def clean_data(df):

# Count invalid entries and show before and after cleaning
    before_cleaning = df.isnull().sum()
    print("Before cleaning:\n", before_cleaning)

# Drop rows where data values aren't mapped correctly eg: NaN
    arr = ["term", "loan_amnt", "issue_d", "last_pymnt_d"]
    for col in arr:
        if col in df.columns:
            df = df.dropna(subset=[col])

# Removing the word "months" from the term column and converting to integer
    if "term" in df.columns:
        df["term"] = df["term"].astype(str).str.replace(" months", "", regex=False).astype(int)

# Simplifying the emp_length column and converting column to integers
    if "emp_length" in df.columns:
        df["emp_length"] = (
            df["emp_length"]
            .str.replace("< 1 year", "1", regex=False)
            .str.replace("10+ years", "10", regex=False)
            .str.replace(" years", "", regex=False)
            .str.replace(" year", "", regex=False)
            .replace(float("NaN"), "0")
            .astype(int)

        )
        df["emp_length"] = df["emp_length"].astype(float).fillna(0).astype(int)

# If interest rate is null, replace with mean interest rate
    if "int_rate" in df.columns:
        df["int_rate"] = df["int_rate"].replace(float("NaN"), df["int_rate"].mean())

# Remove issue_date and last_payment_date columns after calculating payment length
    df = df.drop(columns=["issue_d", "last_pymnt_d"])

# Outputting the result of cleaning the data
    after_cleaning = df.isnull().sum()
    print("After cleaning:\n", after_cleaning)

    return df


def process_data(df, numeric, categorical):
    X = df[numeric + categorical]

    # Generate the mean 
    preprocess = ColumnTransformer([
        # numerical values: standard scaling
        ('num', StandardScaler(), numeric),
        # categorical values/strings: one-hot encoding
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ])

    Xt = preprocess.fit_transform(X)
    print(Xt[0:5])
    return Xt


# Train a machine learning model using the global dataframe and return its output
def train_model(Xt, df, numeric):
    pass

if __name__ == "__main__":
    # Configure pandas
    pd.set_option('display.max_columns', None)
    # Specifying quantitative and qualitative data columns 

    file_path = "loan_data.csv"
    
    # change emp_length and term to numeric from categorical
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
    cleaned_data = clean_data(data)
    print(cleaned_data.head(25))

    # preprocess = process_data(cleaned_data, numeric, categorical)
    # output = train_model(preprocess, data, numeric)