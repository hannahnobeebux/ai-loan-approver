# CHANGELOG.md

This file documents the changes of the project through additions, removals, changes, bugs, and fixes. Changes are documented through branches, user stories, and dates.

## [Unreleased]

### Features

- A CSV file called "loan_data.csv" holds thousands of records that can be used for training the model. It includes information on anonymous financial history and useful values that are used in loan applications such as loan amounts, payback dates and more.
- Purpose of training-data.py: To load, clean and process the data from the CSV file to ensure it's suitable to be used as training data for the model. Then, it will analyse the trained data. 

## [0.3] - 28-10-2025

### Additions

- Created a new file `model.py` which is a class to handle the loading, cleaning, processing, training, and testing of the ML model, as well as accept applications, using the Logistic Regression model.
- Added a logging logic to capture the outputs of each stage within the Model class.
- Removed training-data.py and irrelevant csv data file.

## [0.2] - 24-10-2025

### Additions

- Created the `clean_data` method to start off with converting "term" and "emp_length" columns into integers, so that they can be standardised in the next method. 
- Counting the number of invalid entries (ie those with null values) and also the number of entries after cleaning to visualise the difference 
- Dropping rows with invalid entries. 
- Remove issue_date and last_payment_date columns after calculating payment length

## [0.1] - 21-10-2025

### Additions

- Found a data source (CSV file) for training the data model. 
- Implemented basic data cleaning process by limiting columns needed from CSV file. 
- Using Pandas to create the `load_training_data` method.
- Using Pandas to create the `process_data` method. 
- Processed the "Date" column to be in the right Pandas date format before turning into numerical format to use StandardScalar. 
- Added new column "payment_length_months".

## Template
## [0.0] - Date

### Additions

- x

### Removals

- x

### Changes

- x

### Bugs

- x

### Fixes

- x
