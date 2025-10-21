import pandas as pd

def load_training_data(file_path):
    # pd.set_option('display.max_columns', None)
    # data = pd.read_csv(file_path, 
    #                    names=["Status of existing checking account", "Duration", 
    #                           "Credit history", "Purpose", "Credit amount", "Savings account/bonds", 
    #                           "Present employment since", "Installment rate in percentage of disposable income", 
    #                           "Personal status", "Other debtors/guarantors", "Residence since", "Property", "Age", 
    #                           "Other installment plans", "Housing", "Number of existing credits at this bank", "Job", 
    #                           "Number of people being liable to provide maintenance for", "Telephone", "Foreign worker", 
    #                           "Good/Bad"],
    #                     sep=' ', 
    #                     header=None)
    # print(data.head())
    # return data
    data = pd.read_csv(file_path)
    print(data.head())
    return data


if __name__ == "__main__":
    # Download latest version
    file_path = "credit_risk_dataset.csv"
    print("Path to dataset files:", file_path)
    # file_path = "german-credit-data/german.data"
    load_training_data(file_path)
