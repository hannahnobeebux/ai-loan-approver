from typing import Any, Dict, Optional
from model import Model
import numpy as np
from typing import List
import logging
import pandas as pd

class Decision:
    def __init__(self, outcome: str, ml_score: int, symbolic_score: int, reasons: List[str]) -> None:
        self.outcome = outcome
        self.ml_score = ml_score
        self.symbolic_score = symbolic_score
        self.reasons = reasons  

# SYMBOLIC LAYER
class RuleScorer:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:

        self.score_ml = 0

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
        self.logger.info("INITIALISING RULE SCORER CLASS")

        defaults = {
            # in between deny_threshold and approve_threshold will trigger additional info to be checked
            # eg: 900 is 10% default risk
            "deny_threshold": 700,
            "approve_threshold": 900,

            # employment - penalty if self employed, bonus if full-time employed
            "employment_penalty_self_employed": -20.0,
            "employment_bonus_fte": 10.0,            

            # ask applicant for number of children under 18
            # _ge4 means "greater than or equal to 4"
            # having dependents affecting likelihood of repayment
            "children_penalties_u18": {0: 20, 1: 5, 2: -5, 3: -10},
            "children_penalty_u18_ge4": -15.0,

            # assets bonus points
            # 5pts for each 10k but cap at 80pts
            "assets_bonus_per_10k": 5.0,
            "assets_bonus_cap": 40.0,

            # debt-to-income ratio (DTI) penalty
            # the lower the DTI, the better the chance of repayment
            "dti_penalty_per_point": -5.0,
            "dti_threshold": 10.0,

            "cr_line_duration_considerations": {20: 40, 10: 20, 5: 15, 2: 10},

            "bankruptcy_penalty": -50.0,

        }
        self.cfg = {**defaults, **(config or {})}

    def set_score_ml(self, score: float) -> None:
        self.score_ml = score

    def penalise_children_u18(self, num_children: int) -> int:
        if num_children is None or np.isnan(num_children):
            return 0.0
        n = int(num_children)
        if (n == 1):
            return self.cfg["children_penalties_u18"][1]
        if (n >= 4):
            return self.cfg["children_penalties_u18_ge4"]
        return float(self.cfg["children_penalties_u18"].get(n, 0.0))

    def salary_adj(self, salary: float) -> float:
        if salary is None or np.isnan(salary):
            return 0.0
        n = int(salary // 10000)
        bonus = min(n * self.cfg["salary_bonus_per_10k"], self.cfg["salary_bonus_cap"])
        return float(bonus)

    def assets_adj(self, assets: float) -> float:
        if assets is None or np.isnan(assets):
            return 0.0
        # calculate bonus points for assets
        bonus = min(assets // 10000 * self.cfg["assets_bonus_per_10k"], self.cfg["assets_bonus_cap"])
        return float(bonus)

    def dti_penalty(self, dti: float) -> float:
        if dti is None or np.isnan(dti):
            return 0.0
        if dti > self.cfg["dti_threshold"]:
            return float(dti - self.cfg["dti_threshold"]) * self.cfg["dti_penalty_per_point"]
        return 0.0

    def salary_and_assets_adj(self, salary: float, assets: float) -> float:
        salary_points = self.salary_adj(salary)
        assets_points = self.assets_adj(assets)
        return salary_points + assets_points

    def employment_adj(self, employment_type: Optional[str]) -> float:
        if not employment_type: return 0.0
        et = str(employment_type).strip().lower()
        if "self" in et or "contract" in et:   # self-employed / contractor
            return self.cfg["employment_penalty_self_employed"]
        if "full" in et or "fte" in et:
            return self.cfg["employment_bonus_fte"]
        return 0.0
    
    def cr_line_duration_adj(self, years: float) -> float:
        # enumerate through entire dictionary to find a satisfactory value for the score adjustment
        for (key, val) in self.cfg["cr_line_duration_considerations"].items():
            if years >= int(key):
                return int(val)
        # if below the lowest value, return the lowest adjustment (most negative)
        return -15.0
    
    def bankruptcy_penalty(self, experienced_bankruptcy: bool) -> float:
        if experienced_bankruptcy:
            return self.cfg["bankruptcy_penalty"]
        return 0.0

        
# amend credit risk after decision rules
# will go through each rule and apply adjustments
# this is just the symbolic part and runs through all cases regardless of ML score 
    def apply_all_rules(self, sample_applicant, applicant_info) -> float:
        self.logger.info(f"Applying all rules to applicant: {applicant_info.get('id')}")
        adjustment = 0.0
        adjustment += self.assets_adj(applicant_info.get("assets", 0.0))
        adjustment += self.salary_adj(applicant_info.get("annual_income", 0.0))
        adjustment += self.penalise_children_u18(applicant_info.get("num_children_u18", 0))
        adjustment += self.employment_adj(applicant_info.get("employment_type", ""))
        adjustment += self.dti_penalty(applicant_info.get("dti", 0.0))
        new_score = self.score_ml + adjustment
        self.logger.info(f"Original ML Score: {self.score_ml}, Adjustment: {adjustment}, New Score: {new_score}")
        # return new_score

# this method adjust ML score depending on the least riskiest to most riskiest considerations
    def adjust_score_and_decide(self, applicant_info) -> Decision:
        # take the risk score from the model component
        acc_score = 0; 
        pos_reasons = []
        neg_reasons = []

        if (self.score_ml >= 950):
            return Decision(
                outcome="APPROVE",
                ml_score=self.score_ml,
                symbolic_score=self.score_ml,
                reasons=[f"ML score {self.score_ml}. Approved without adjustments."]
            )
        
        # start with least riskiest - going from lowest to highest ML score
        # if you have a 20-25% chance of defaulting - deny right away. the ML model is based off historical data - an accurate representation of risk
        # score needs to be above 700 but below 950 to consider adjustments

        if (self.score_ml <= 700):
            return Decision(
                outcome="DENY",
                ml_score=self.score_ml,
                symbolic_score=self.score_ml,
                reasons=[f"ML score {self.score_ml}. Denied without adjustments."]
            )
        
        # riskiest = assets
        if (self.score_ml <= 750):
            temp = acc_score
            acc_score += self.assets_adj(applicant_info.get("assets", 0.0))
            if acc_score < temp:
                neg_reasons.append("Assets caused risk score to increase.")
            else:
                pos_reasons.append("Assets caused risk score to decrease.")

        if (self.score_ml <= 800):
            temp = acc_score
            acc_score += self.employment_adj(applicant_info.get("employment_type", ""))
            if acc_score < temp:
                neg_reasons.append("Employment type caused risk score to increase.")
            else:
                pos_reasons.append("Employment type caused risk score to decrease.")

        if (self.score_ml <= 850):
            temp = acc_score
            acc_score += self.penalise_children_u18(applicant_info.get("num_children_u18", 0))
            acc_score += self.dti_penalty(applicant_info.get("dti", 0.0))
            if acc_score < temp:
                neg_reasons.append("Children and debt-to-income ration caused risk score to increase.")
            else:
                pos_reasons.append("Children and debt-to-income ratio caused risk score to decrease.")

        if (self.score_ml <= 900):
            temp = acc_score
            acc_score += self.cr_line_duration_adj(applicant_info.get("cr_line_duration_years", 0))
            if acc_score < temp:
                neg_reasons.append("Credit line duration caused risk score to increase.")
            else:
                pos_reasons.append("Credit line duration caused risk score to decrease.")

        if (self.score_ml < 950):
            temp = acc_score
            acc_score += self.bankruptcy_penalty(applicant_info.get("experienced_bankruptcy", False))
            if acc_score < temp:
                neg_reasons.append("Bankruptcy history caused risk score to increase.")


    #    using new score     
        if (acc_score < 0):
            return Decision(
                outcome="DENY",
                ml_score=self.score_ml,
                symbolic_score=acc_score,
                reasons=", ".join(neg_reasons)
            )
        else:
            return Decision(
                outcome="APPROVE",
                ml_score=self.score_ml,
                symbolic_score=acc_score,
                reasons=", ".join(pos_reasons)
            )


class LogicComponent():
    def __init__(self):
        
        self.model = Model()

        self.numeric = [
            "loan_amnt", "int_rate",
            "annual_inc", "delinq_2yrs", "open_acc", "pub_rec",
            "revol_bal", "repay_fail", "term"
        ]
        self.categorical = [
            "emp_length", "home_ownership",
            "purpose"
        ]

        self.filename = "loan_data.csv"
        self.nrows = 10000
        self.rules = RuleScorer()

    def use_ml_model(self, sample_applicant: pd.DataFrame) -> None:
        self.model.load_data(self.filename, self.numeric, self.categorical, self.nrows)
        self.model.train_model()
        self.model.test_model()  # logs your metrics    
        score = self.model.process_application(sample_applicant)
        return score 

    def make_decision(self, sample_applicant: pd.DataFrame, applicant_info: Dict[str, Any]) -> Decision:
        score = self.use_ml_model(sample_applicant)
        self.rules.set_score_ml(score)
        decision = self.rules.adjust_score_and_decide(applicant_info)
        return decision



if __name__ == '__main__':

    # pass in sample application - coming from GUI in the form of a JSON
    # pass in extra info such as number of children, assets, employment type, etc.

    # using logic.decide() to return approve/deny
    # if in betweeen threshholds - in review - extra function which checks if salary is high enough etc..
    # mvp = in between thresholds - deny 

    sample_applicant = pd.DataFrame([{
        'loan_amnt': 6000.0,                # requested loan amount
        'term': 36,                         # loan term in months
        'int_rate': 12.5,                   # offered APR %
        'emp_length': 6,                    # years employed (or bucketed int if that's how you encoded)
        'home_ownership': 'RENT',           # home ownership category
        'annual_inc': 55000.0,              # yearly income
        'purpose': 'debt_consolidation',    # loan purpose category
        'delinq_2yrs': 0.0,                 # past 2y delinquencies
        'open_acc': 6.0,                    # number of open credit lines
        'pub_rec': 0.0,                     # public derogatories
        'revol_bal': 3200.0                # revolving balance
    }])

# make a file to hold sample applicants - as a .py file and export them as pandas dataframes 
    # HIGH RISK 

    # MEDIUM RISK 

    # LOW RISK 

    applicant_info = {
        "employment_type": "Full-time",
        "num_children_u18": 0,
        "assets": 25000.0,
        "dti": 0.1,
        "cr_line_duration_years": 5,  # years with credit line
        "experienced_bankruptcy": False,
    }

    logic = LogicComponent()

    decision = logic.make_decision(sample_applicant, applicant_info=applicant_info)
    print(f"Decision Outcome: {decision.outcome}, ML Score: {decision.ml_score}, Symbolic Score: {decision.symbolic_score}, Reasons: {decision.reasons}")


