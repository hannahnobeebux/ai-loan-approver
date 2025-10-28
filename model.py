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

# A class used to represent a machine-learning model using Logistic Regression
class Model:

    def __init__(self):
        pd.set_option('display.max_columns', None)
        self.data = []
        self.numeric_cols = []
        self.categorical_cols = []
        self.X_train, self.y_train, self.X_cal, self.y_cal, self.X_test, self.y_test = (None,)*6
        self.pipeline, self.model = (None,)*2

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
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        cols = numeric_cols + categorical_cols
        try:
            self.data = pd.read_csv(train_data_filepath, nrows=nrows, usecols=cols)
            self.logger.info(f"Successfully Loaded Data:\n(first 5 rows)\n{self.data.head(5)}")
        except:
            self.logger.error("Loading Data Failed!")

    def clean_data(self):
        self.logger.info("Cleaning Data...")
        # Before Cleaning
        self.logger.info(f"Before Cleaning (null count):\n{self.data.isnull().sum()}")

        # Convert strings into date formats
        date_format = "%b-%y"
        self.convert_date_col("issue_d", date_format)
        self.convert_date_col("last_pymnt_d", date_format)

        # Calculate payment length in months
        self.data["payment_length_months"] = ((self.data["last_pymnt_d"].dt.year - self.data["issue_d"].dt.year) * 12 + 
                                                    (self.data["last_pymnt_d"].dt.month - self.data["issue_d"].dt.month))
        
        self.data = self.data.drop(columns=["last_pymnt_d", "issue_d"])
        self.categorical_cols.remove("issue_d")
        self.categorical_cols.remove("last_pymnt_d")
        self.numeric_cols.append("payment_length_months")
        self.numeric_cols.append("term")
        self.categorical_cols.remove("term")

        # Drop rows where data values aren't mapped correctly eg: NaN
        arr = ["term", "loan_amnt", "issue_d", "last_pymnt_d", "payment_length_months"]
        for col in arr:
            if col in self.data.columns:
                self.data = self.data.dropna(subset=[col])
        
        # Removing the word "months" from the term column and converting to integer
        if "term" in self.data.columns:
            self.data["term"] = self.data["term"].astype(str).str.replace(" months", "", regex=False).astype(int)

        # Simplifying the emp_length column and converting column to integers
        if "emp_length" in self.data.columns:
            self.data["emp_length"] = (
                self.data["emp_length"]
                .str.replace("< 1 year", "1", regex=False)
                .str.replace("10+ years", "10", regex=False)
                .str.replace(" years", "", regex=False)
                .str.replace(" year", "", regex=False)
                .replace(float("NaN"), "0")
                .astype(int)
            )
            self.data["emp_length"] = self.data["emp_length"].astype(float).fillna(0).astype(int)

        # If interest rate is null, replace with mean interest rate
        # if "int_rate" in self.data.columns:
        #     self.data["int_rate"] = self.data["int_rate"].replace(float("NaN"), self.data["int_rate"].mean())
        apr_mean = self.data["int_rate"].mean()
        if "int_rate" in self.data.columns:
            self.data["int_rate"] = self.data["int_rate"].fillna(apr_mean)

        # Handling outliers by capping and log transforming positively skewed data
 
        for col in ["loan_amnt", "annual_inc", "revol_bal"]:
            if col in self.data.columns:
                upper = self.data[col].quantile(0.99)
                self.data[col] = np.clip(self.data[col], 0, upper)
                # or log transform if positively skewed:
                self.data[col] = np.log1p(self.data[col])
        
        # After Cleaning
        self.logger.info(f"After Cleaning (null count):\n{self.data.isnull().sum()}")
        self.logger.info(f"Successfully Cleaned Data:\n(first 5 rows)\n{self.data.head(5)}")
        # self.logger.info(f"Number of rows after cleaning: {len(self.data)}")

    def convert_date_col(self, col_name, date_format):
        self.data[col_name] = pd.to_datetime(self.data[col_name], format=date_format)

    def process_data(self):
        self.logger.info("Processing Data...")
        y = self.data["repay_fail"]
        # Obtain data into 2 buckets: training & calibration + testing
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(self.data.drop(columns="repay_fail"), y, 
                                                            test_size=0.4, random_state=42, stratify=y)
        self.X_cal, self.X_test, self.y_cal, self.y_test = train_test_split(X_temp, y_temp, 
                                                            test_size=0.5, random_state=42, stratify=y_temp)
        self.numeric_cols.remove("repay_fail")
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), self.numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_cols),
        ])
        lr = LogisticRegression(max_iter=1000)
        self.pipeline = Pipeline([("prep", preprocessor), ("model", lr)])
        data = preprocessor.fit_transform(self.X_train, self.y_train)
        self.logger.info(f"First 5 Rows of pre-processed data:\n{data[0:5]}")

    def train_model(self):
        self.logger.info("Training Model...")
        self.clean_data()
        self.process_data()
        self.pipeline.fit(self.X_train, self.y_train)
        self.model = CalibratedClassifierCV(
            base_estimator=self.pipeline,
            cv="prefit",
            method="isotonic"
        )
        self.model.fit(self.X_cal, self.y_cal)
        self.logger.info("Model Trained Successfully!")

    def test_model(self):
        self.logger.info("Testing Model...")
        y_proba = self.model.predict_proba(self.Xt)[:,1]
        self.logger.info(f"ROC-AUC:\n{roc_auc_score(self.yt, y_proba)}")
        self.logger.info(f"PR_AUC:\n{average_precision_score(self.yt, y_proba)}")

        cutoffs = [0.05, 0.08, 0.12, 0.16]
        for cutoff in cutoffs:
            y_pred_policy = (y_proba >= cutoff).astype(int)  # 1 = predict default (reject)
            self.logger.info(f"Confusion Matrix (reject=1 at cutoff {cutoff}):\n{confusion_matrix(self.yt, y_pred_policy)}")
            self.logger.info(classification_report(self.yt, y_pred_policy, digits=3))
        self.logger.info("Model Tested Successfully!")

    def process_application(self, X):
        probability = self.model.predict_proba(X)
        risk_score = 1000 - (probability * 1000)
        return risk_score