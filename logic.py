from typing import Any, Dict, Optional
from model import Model
import numpy as np
from typing import List
import logging
import pandas as pd

class Decision:
    outcome: str
    ml_score: int    
    symbolic_score: int
    reasons: List[str]

# SYMBOLIC LAYER
class RuleScorer:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:

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
            "employment_penalty_self_employed": -40.0,
            "employment_bonus_fte": +10.0,            

            # ask applicant for number of children under 18
            # _ge4 means "greater than or equal to 4"
            # having dependents affecting likelihood of repayment
            "children_penalties_u18": {0: 0, 1: -20, 2: -50, 3: -100},
            "children_penalty_u18_ge4": -150.0,

            # assets bonus points
            # 5pts for each 10k but cap at 80pts
            "assets_bonus_per_10k": 5.0,
            "assets_bonus_cap": 80.0,

            # debt-to-income ratio (DTI) penalty
            # the lower the DTI, the better the chance of repayment
            "dti_penalty_per_point": -5.0,
            "dti_threshold": 20.0,
        }
        self.cfg = {**defaults, **(config or {})}

    def penalise_children_u18(self, num_children: int) -> int:
        if num_children is None or np.isnan(num_children):
            return 0.0
        n = int(num_children)
        if n >= 4:
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
        
# amend credit risk after decision rules
# will go through each rule and apply adjustments
    def apply_all_rules(self, sample_applicant, applicant_info) -> float:
        self.logger.info(f"Applying all rules to applicant: {applicant_info.get('id')}")
        score_ml = int(self.model.process_application(sample_applicant))
        adjustment = 0.0
        adjustment += self.assets_adj(applicant_info.get("assets", 0.0))
        adjustment += self.salary_adj(applicant_info.get("annual_income", 0.0))
        adjustment += self.penalise_children_u18(applicant_info.get("num_children_u18", 0))
        adjustment += self._employment_adj(applicant_info.get("employment_type", ""))
        adjustment += self.dti_penalty(applicant_info.get("dti", 0.0))
        new_score = score_ml + adjustment
        self.logger.info(f"Original ML Score: {score_ml}, Adjustment: {adjustment}, New Score: {new_score}")
        # return new_score

# this method adjust ML score depending on the least riskiest to most riskiest considerations
    def adjust_score_and_decide(self, sample_applicant, applicant_info) -> Decision:
        # take the risk score from the model component
        score_ml = int(self.model.process_application(sample_applicant))
        acc_score = 0; 
        pos_reasons = []
        neg_reasons = []

        if (score_ml >= 950):
            return Decision(
                outcome="APPROVE",
                ml_score=score_ml,
                symbolic_score=score_ml,
                reasons=[f"ML score {score_ml}. Approved without adjustments."]
            )
        
        # start with least riskiest - going from lowest to highest ML score
        # if you have a 20-25% chance of defaulting - deny right away. the ML model is based off historical data - an accurate representation of risk
        # score needs to be above 700 but below 950 to consider adjustments

        if (score_ml <= 700):
            return Decision(
                outcome="DENY",
                ml_score=score_ml,
                symbolic_score=score_ml,
                reasons=[f"ML score {score_ml}. Denied without adjustments."]
            )
        
        # riskiest = assets
        if (score_ml <= 750):
            temp = acc_score
            acc_score += self.assets_adj(applicant_info.get("assets", 0.0))
            if acc_score < temp:
                neg_reasons.append("Assets caused risk score to increase.")
            else:
                pos_reasons.append("Assets caused risk score to decrease.")
        
        if (score_ml <= 800):
            acc_score += self.employment_adj(applicant_info.get("employment_type", ""))
            if acc_score < temp:
                neg_reasons.append("Employment type caused risk score to increase.")
            else:
                pos_reasons.append("Employment type caused risk score to decrease.")

        if (score_ml <= 850):
            acc_score += self.penalise_children_u18(applicant_info.get("num_children_u18", 0))
            acc_score += self.dti_penalty(applicant_info.get("dti", 0.0))
            if acc_score < temp:
                neg_reasons.append("Children and debt-to-income ration caused risk score to increase.")
            else:
                pos_reasons.append("Children and debt-to-income ratio caused risk score to decrease.")

    #    using new score     
        if (acc_score < 0):
            return Decision(
                outcome="DENY",
                ml_score=score_ml,
                symbolic_score=acc_score,
                reasons=neg_reasons.join(", ")
            )
        else:
            return Decision(
                outcome="APPROVE",
                ml_score=score_ml,
                symbolic_score=acc_score,
                reasons=pos_reasons.join(", ")
            )


class LogicComponent():
    def __init__(self):
        
        self.model = Model()
        self.rules = RuleScorer()

        self.numeric = [
            "loan_amnt", "int_rate",
            "annual_inc", "delinq_2yrs", "open_acc", "pub_rec",
            "revol_bal", "repay_fail"
        ]
        self.categorical = [
            "term", "emp_length", "home_ownership", "issue_d",
            "purpose", "last_pymnt_d"
        ]

        # self.model.load_data("loan_data.csv", self.numeric, self.categorical, 2000)
        # self.model.train_model()
        # self.model.test_model()
        
    def use_ml_model(self, csv_path: str, nrows: Optional[int] = None) -> None:
        self.model.load_data(csv_path, self.numeric, self.categorical, nrows)
        self.model.train_model()
        self.model.test_model()  # logs your metrics    
        
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
        'revol_bal': 3200.0,                # revolving balance

    }])

        score = self.model.process_application(sample_applicant)
        return score 


if __name__ == '__main__':
    logic = LogicComponent()

    # pass in sample application - coming from GUI in the form of a JSON
    # pass in extra info such as number of children, assets, employment type, etc.

    # using logic.decide() to return approve/deny
    # if in betweeen threshholds - in review - extra function which checks if salary is high enough etc..
    # mvp = in between thresholds - deny 

    logic = LogicComponent()
    logic.use_ml_model("loan_data.csv", nrows=2000)

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

    applicant_info = {
        "employment_type": "Full-time",
        "num_children_u18": 2,
        "assets": 25000.0,
        "dti": 18.0
    }