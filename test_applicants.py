import pandas as pd

# The below tests are for applicants who will get varying generated risk scores from the ML model
# GUI acts as E2E tests - uses both ML score and Symbolic reasons

# HIGH RISK APPLICANT
high_risk_applicant = pd.DataFrame([{
    'loan_amnt': 35000.0,               # large loan amount
    'term': 60,                         # longer term
    'int_rate': 18.5,                   # high interest rate
    'emp_length': 1,                    # short employment history
    'home_ownership': 'RENT',           # renting
    'annual_inc': 28000.0,              # low income
    'purpose': 'other',                 # risky purpose
    'delinq_2yrs': 3.0,                 # multiple delinquencies
    'open_acc': 15.0,                   # many open accounts
    'pub_rec': 1.0,                     # public record
    'revol_bal': 8500.0                 # high revolving balance
}])

high_risk_info = {
    "employment_type": "Contract",       # contract work (penalty)
    "num_children_u18": 4,              # many dependents
    "assets": 500.0,                    # minimal assets
    "dti": 0.45,                        # high debt-to-income
    "cr_line_duration_years": 1,        # short credit history
    "experienced_bankruptcy": True,      # bankruptcy history
    "age": 30
}

# MEDIUM RISK APPLICANT
medium_risk_applicant = pd.DataFrame([{
    'loan_amnt': 15000.0,               # moderate loan amount
    'term': 36,                         # standard term
    'int_rate': 14.2,                   # moderate interest rate
    'emp_length': 4,                    # decent employment history
    'home_ownership': 'MORTGAGE',       # homeowner
    'annual_inc': 45000.0,              # moderate income
    'purpose': 'debt_consolidation',    # reasonable purpose
    'delinq_2yrs': 1.0,                 # one delinquency
    'open_acc': 8.0,                    # reasonable open accounts
    'pub_rec': 0.0,                     # no public records
    'revol_bal': 4200.0                 # moderate revolving balance
}])

medium_risk_info = {
    "employment_type": "Part-time",      # part-time work
    "num_children_u18": 2,              # moderate dependents
    "assets": 8000.0,                   # some assets
    "dti": 0.25,                        # moderate debt-to-income
    "cr_line_duration_years": 5,        # decent credit history
    "experienced_bankruptcy": False,     # no bankruptcy
    "age": 45
}

# LOW RISK APPLICANT
low_risk_applicant = pd.DataFrame([{
    'loan_amnt': 8000.0,                # smaller loan amount
    'term': 36,                         # standard term
    'int_rate': 10.8,                   # low interest rate
    'emp_length': 8,                    # long employment history
    'home_ownership': 'OWN',            # owns home
    'annual_inc': 75000.0,              # high income
    'purpose': 'home_improvement',      # good purpose
    'delinq_2yrs': 0.0,                 # no delinquencies
    'open_acc': 5.0,                    # few open accounts
    'pub_rec': 0.0,                     # no public records
    'revol_bal': 1200.0                 # low revolving balance
}])

low_risk_info = {
    "employment_type": "Full-time",      # stable employment (bonus)
    "num_children_u18": 1,              # one dependent
    "assets": 45000.0,                  # substantial assets
    "dti": 0.08,                        # low debt-to-income
    "cr_line_duration_years": 15,       # long credit history
    "experienced_bankruptcy": False,     # no bankruptcy
    "age": 18
}

# EDGE CASE: VERY HIGH SCORE APPLICANT (should auto-approve)
excellent_applicant = pd.DataFrame([{
    'loan_amnt': 5000.0,                # small loan
    'term': 24,                         # short term
    'int_rate': 8.5,                    # excellent rate
    'emp_length': 10,                   # very long employment
    'home_ownership': 'OWN',            # owns home
    'annual_inc': 95000.0,              # high income
    'purpose': 'medical',               # necessary purpose
    'delinq_2yrs': 0.0,                 # perfect record
    'open_acc': 3.0,                    # minimal accounts
    'pub_rec': 0.0,                     # clean record
    'revol_bal': 200.0                  # very low balance
}])

excellent_info = {
    "employment_type": "Full-time",
    "num_children_u18": 0,
    "assets": 75000.0,
    "dti": 0.05,
    "cr_line_duration_years": 20,
    "experienced_bankruptcy": False,
    "age": 23
}

# EDGE CASE: VERY LOW SCORE APPLICANT (should auto-deny)
poor_applicant = pd.DataFrame([{
    'loan_amnt': 40000.0,               # very large loan
    'term': 60,                         # long term
    'int_rate': 25.0,                   # very high rate
    'emp_length': 0,                    # unemployed/new job
    'home_ownership': 'RENT',           # renting
    'annual_inc': 15000.0,              # very low income
    'purpose': 'other',                 # vague purpose
    'delinq_2yrs': 5.0,                 # many delinquencies
    'open_acc': 20.0,                   # too many accounts
    'pub_rec': 3.0,                     # multiple public records
    'revol_bal': 15000.0                # maxed out credit
}])

poor_info = {
    "employment_type": "Unemployed",
    "num_children_u18": 5,
    "assets": 0.0,
    "dti": 0.8,
    "cr_line_duration_years": 0.5,
    "experienced_bankruptcy": True,
    "age": 45
}

# Below test cases will use hardcoded ML scores just to test symbolic layer in isolation
# Test case: Good ML Score but bad reasons for a symbolic score --> outcome = DENY eg: bankruptcy or age
# Test case: mix of good and bad reasons but gets denied overall
# Test case: mix of good and bad reasons but gets approved overall

# Test case: Bad ML Score but good reasons for a symbolic score --> outcome = APPROVE
bad_ml_good_symbolic_info = {
    "employment_type": "Full-time",      # +10 bonus
    "num_children_u18": 0,              # no penalty
    "assets": 100000.0,                 # +50 bonus (high assets)
    "dti": 0.10,                        # low DTI, no penalty
    "cr_line_duration_years": 25,       # +15 bonus (long credit history)
    "experienced_bankruptcy": False,     # no penalty
    "age": 35                           # no penalty
}

# Failing test - incorrect passing of hardcoded ML score 


# List of all test cases for easy iteration
test_cases = [
    ("High Risk", high_risk_applicant, high_risk_info, None),
    ("Medium Risk", medium_risk_applicant, medium_risk_info, None),
    ("Low Risk", low_risk_applicant, low_risk_info, None),
    ("Excellent", excellent_applicant, excellent_info, None),
    ("Poor", poor_applicant, poor_info, None), 
    ("Hardcoded ML | Poor Credit Risk + Good factors", {}, bad_ml_good_symbolic_info, 701)
]

