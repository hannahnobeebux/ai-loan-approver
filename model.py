import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import logging

# A class used to represent the ML model to produce a credit risk score
class Model:

    def __init__(self):
        pd.set_option('display.max_columns', None)
        # Initialisation of class attributes to be used between methods
        self.data = []
        self.numeric_cols = []
        self.categorical_cols = []
        self.X_train, self.y_train, self.X_cal, self.y_cal, self.X_test, self.y_test = (None,)*6
        self.pipeline, self.model = (None,)*2

        # Initialise logger with certain format
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename="logs/model.log",
            encoding="utf-8",
            format="{asctime} - {levelname} - {message}\n",
            filemode="w",
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.DEBUG
        )
        self.logger.info("INITIALISING MODEL CLASS")

    def load_data(self, train_data_filepath, numeric_cols, categorical_cols, nrows):
        self.logger.info("Loading Data...")
        # Define the numeric and categorical columns for later preprocessing
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        cols = numeric_cols + categorical_cols
        # Attempt to read the csv of the chosen number of rows and columns
        try:
            self.data = pd.read_csv(train_data_filepath, nrows=nrows, usecols=cols)
            self.logger.info(f"Successfully Loaded Data:\n(first 5 rows)\n{self.data.head(5)}")
        except:
            self.logger.error("Loading Data Failed!")

    def clean_data(self):
        self.logger.info("Cleaning Data...")
        # Before Cleaning
        self.logger.info(f"Before Cleaning (null count):\n{self.data.isnull().sum()}")

        # Drop rows where data values aren't mapped correctly eg: NaN
        arr = ["term", "loan_amnt"]
        for col in arr:
            if col in self.data.columns:
                self.data = self.data.dropna(subset=[col])
        
        # Removing the word "months" from the term column and convert to integer
        if "term" in self.data.columns:
            self.data["term"] = self.data["term"].astype(str).str.replace(" months", "", regex=False).astype(int)

        # Simplifying the emp_length column and converting column to integers
        if "emp_length" in self.data.columns:
            emp = self.data["emp_length"].astype(str)
            emp = emp.str.replace("< 1 year", "0", regex=False)
            emp = emp.str.replace("10+ years", "10", regex=False)
            emp = emp.str.replace(" years", "", regex=False)
            emp = emp.str.replace(" year", "", regex=False)
            emp = pd.to_numeric(emp, errors="coerce").fillna(0).astype(int)
            self.data["emp_length"] = emp

        # If interest rate is null, replace with mean interest rate
        apr_mean = self.data["int_rate"].mean()
        if "int_rate" in self.data.columns:
            self.data["int_rate"] = self.data["int_rate"].fillna(apr_mean)

        # Handle outliers
        for col in ["loan_amnt", "annual_inc", "revol_bal"]:
            if col in self.data.columns:
                upper = self.data[col].quantile(0.99)
                self.data[col] = np.clip(self.data[col], 0, upper)
        
        # Transform Columns
        if all(c in self.data.columns for c in ["loan_amnt", "annual_inc"]):
            self.data["loan_to_income"] = self.data["loan_amnt"] / (self.data["annual_inc"] + 1)

        if all(c in self.data.columns for c in ["revol_bal", "annual_inc"]):
            self.data["revol_to_income"] = self.data["revol_bal"] / (self.data["annual_inc"] + 1)

        if all(c in self.data.columns for c in ["open_acc", "revol_bal"]):
            self.data["revol_per_acc"] = self.data["revol_bal"] / (self.data["open_acc"] + 1)

        self.numeric_cols.append("loan_to_income")
        self.numeric_cols.append("revol_to_income")
        self.numeric_cols.append("revol_per_acc")
        
        # After Cleaning
        self.logger.info(f"After Cleaning (null count):\n{self.data.isnull().sum()}")
        self.logger.info(f"Successfully Cleaned Data:\n(first 5 rows)\n{self.data.head(5)}")
        self.logger.info(f"Feature summary:\n{self.data.describe(include='all')}")
        # self.logger.info(f"Number of rows after cleaning: {len(self.data)}")

    def process_data(self):
        self.logger.info("Processing Data...")

        # Identify label
        y = self.data["repay_fail"]
        
        # Split data into 3 categories: training, tetsing, calibrating
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(self.data.drop(columns="repay_fail"), y, 
                                                            test_size=0.4, random_state=42, stratify=y)
        self.X_cal, self.X_test, self.y_cal, self.y_test = train_test_split(X_temp, y_temp, 
                                                            test_size=0.5, random_state=42, stratify=y_temp)
        # Remove label
        if "repay_fail" in self.numeric_cols:
            self.numeric_cols.remove("repay_fail")

        # Form pipeline for preprocessing different data types
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), self.numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_cols),
        ])

        # Add ML model
        lr = LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs", class_weight="balanced")
        self.pipeline = Pipeline([("prep", preprocessor), ("model", lr)])

        # Process data
        data = preprocessor.fit_transform(self.X_train, self.y_train)
        self.logger.info(f"First 5 Rows of pre-processed data:\n{data[0:5]}")

    def train_model(self):
        self.logger.info("Training Model...")
        # Handle data cleaning and processing
        self.clean_data()
        self.process_data()

        # Preprocess and fit the data against the model
        self.pipeline.fit(self.X_train, self.y_train)

        # Calibrate the model
        self.model = CalibratedClassifierCV(
            estimator=self.pipeline,
            cv=5,
            method="isotonic"
        )
        self.model.fit(self.X_cal, self.y_cal)
        self.logger.info("Model Trained Successfully!")

    def test_model(self):
        self.logger.info("Testing Model...")
        # Produce risk probabilities (removing the label) and produce accuracy scores
        y_proba = self.model.predict_proba(self.X_test)[:,1]
        self.logger.info(f"ROC-AUC:\n{roc_auc_score(self.y_test, y_proba)}")
        self.logger.info(f"PR_AUC:\n{average_precision_score(self.y_test, y_proba)}")

        # Produce confusion matrices and classification reports for different cutoff values
        cutoffs = [0.05, 0.08, 0.12, 0.16, 0.20, 0.25]
        for cutoff in cutoffs:
            y_pred_policy = (y_proba >= cutoff).astype(int)
            self.logger.info(f"Confusion Matrix (reject=1 at cutoff {cutoff}):\n"
                             f"{confusion_matrix(self.y_test, y_pred_policy)}")
            self.logger.info(f"Classification Report\n{classification_report(self.y_test, y_pred_policy, digits=3)}")
        self.logger.info("Model Tested Successfully!")

    def process_application(self, X):
        # First produce the transformation columns
        if all(c in X.columns for c in ["loan_amnt", "annual_inc"]):
            X["loan_to_income"] = X["loan_amnt"] / (X["annual_inc"] + 1)

        if all(c in X.columns for c in ["revol_bal", "annual_inc"]):
            X["revol_to_income"] = X["revol_bal"] / (X["annual_inc"] + 1)

        if all(c in X.columns for c in ["open_acc", "revol_bal"]):
            X["revol_per_acc"] = X["revol_bal"] / (X["open_acc"] + 1)
        
        # Obtain risk probability
        probability = self.model.predict_proba(X)[:,1][0]

        # Transform risk score into credit risk score and return
        risk_score = 1000 - (probability * 1000)
        print(f"Predicted Application Score = {risk_score}")
        return risk_score
    

if __name__ == '__main__':
    model = Model()
    # Define categories to be chosen from the csv
    numeric = [
        "loan_amnt", "int_rate",
        "annual_inc", "delinq_2yrs", "open_acc", "pub_rec",
        "revol_bal", "repay_fail", "term"
    ]
    categorical = ["emp_length", "home_ownership", "purpose"]

    # Load a specific csv with chosen columns and rows
    model.load_data("loan_data.csv", numeric, categorical, 10000)
    model.train_model()
    model.test_model()

    # Test the model with a sample application
    sample_applicant = pd.DataFrame([{
        'loan_amnt': 6000.0,                # requested loan amount
        'term': 36,                         # loan term in months
        'int_rate': 12.5,                   # offered APR %
        'emp_length': 6,                    # years employed
        'home_ownership': 'RENT',           # home ownership category
        'annual_inc': 55000.0,              # yearly income
        'purpose': 'debt_consolidation',    # loan purpose category
        'delinq_2yrs': 0.0,                 # past 2y delinquencies
        'open_acc': 6.0,                    # number of open credit lines
        'pub_rec': 0.0,                     # public derogatories
        'revol_bal': 3200.0,                # revolving balance

    }])

    # Output the sample applications score
    score = model.process_application(sample_applicant)
    model.logger.info(f"Predicted Application Score = {score}")