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

            "age_considerations": {21: -5}

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
        print(f"years = {years}")
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
    
    def age_penalty(self, age: int) -> float:
        if age is None or np.isnan(age):
            return 0.0
        for (key, val) in self.cfg["age_considerations"].items():
            if age <= int(key):
                return float(val)
        return 0  # catching edge case for below 21 years old


# create new score after decision rules
# will go through each rule and apply adjustments
# this is just the symbolic part and runs through all cases regardless of ML score 

    # testing purpose for all methods
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

    def adjust_score_and_decide(self, applicant_info) -> Decision:
        acc_score = 0; 
        pos_reasons = []
        neg_reasons = []

        if (self.score_ml >= 950):
            return Decision(
                outcome="APPROVE",
                ml_score=self.score_ml,
                symbolic_score=0,
                reasons=[f"ML score {self.score_ml}. Approved without adjustments."]
            )
        
        # start with least riskiest - going from lowest to highest ML score
        # if you have a 20-25% chance of defaulting - deny right away. the ML model is based off historical data - an accurate representation of risk
        # score needs to be above 700 but below 950 to consider adjustments

        if (self.score_ml <= 700):
            return Decision(
                outcome="DENY",
                ml_score=self.score_ml,
                symbolic_score=0,
                reasons=[f"ML score {self.score_ml}. Denied without adjustments."]
            )
        
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
            temp2 = acc_score
            acc_score += self.age_penalty(applicant_info.get("age", 0))
            if acc_score < temp2:
                neg_reasons.append("Age caused risk score to increase.")
            else:
                pos_reasons.append("Age caused risk score to decrease.")


    # using new score
        if (acc_score < 0):
            return Decision(
                outcome="DENY",
                ml_score=self.score_ml,
                symbolic_score=acc_score,
                reasons=neg_reasons
            )
        else:
            return Decision(
                outcome="APPROVE",
                ml_score=self.score_ml,
                symbolic_score=acc_score,
                reasons=pos_reasons
            )
        


class LogicComponent():
    def __init__(self):

        self.logger = logging.getLogger("LogicComponent")

        log_handler = logging.FileHandler("logs/logic.log", mode="w")
        log_formatter = logging.Formatter(
            "{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        log_handler.setFormatter(log_formatter)
        self.logger.addHandler(log_handler)
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.info("="*60)
        self.logger.info("INITIALISING LOGIC COMPONENT")
        self.logger.info("="*60)
        
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
        self.logger.info("-" * 40)
        self.logger.info("Loading and training ML model...")
        self.model.load_data(self.filename, self.numeric, self.categorical, self.nrows)
        self.model.train_model()
        self.model.test_model()  
        score = self.model.process_application(sample_applicant)
        self.logger.info(f"ML model returned score: {score}")
        self.logger.info("-" * 40)
        return score 

    def make_decision(self, sample_applicant: pd.DataFrame, applicant_info: Dict[str, Any], hardcoded_ml_score = None) -> Decision:
        self.logger.info(f"\nProcessing applicant with info:")
        self.logger.info(f"  Employment: {applicant_info.get('employment_type', 'N/A')}")
        self.logger.info(f"  Children: {applicant_info.get('num_children_u18', 'N/A')}")
        self.logger.info(f"  Assets: ${applicant_info.get('assets', 'N/A'):,.2f}")
        self.logger.info(f"  DTI: {applicant_info.get('dti', 'N/A')}")
        self.logger.info(f"  Age: {applicant_info.get('age', 'N/A')}")


        score = self.use_ml_model(sample_applicant) if hardcoded_ml_score is None else hardcoded_ml_score
        self.rules.set_score_ml(score)
        decision = self.rules.adjust_score_and_decide(applicant_info)
        
        self.logger.info(f"\nDECISION SUMMARY:")
        self.logger.info(f"  Outcome: {decision.outcome}")
        self.logger.info(f"  ML Score: {decision.ml_score:.2f}")
        self.logger.info(f"  Symbolic Score: {decision.symbolic_score:.2f}")
        self.logger.info(f"  Reasons: {'\n'.join(decision.reasons)}")        
        
        return decision

    def process_application(self, applicant: Dict[str, Any], hardcode_ml_score = None) -> Decision:
        if hardcode_ml_score is not None:
            self.rules.set_score_ml(hardcode_ml_score)
        else:
            ml_score = self.use_ml_model(pd.DataFrame([applicant]))
            self.rules.set_score_ml(ml_score)

        symbolic_applicant_info = {
            "employment_type": applicant.get("employment_type", ""),
            "num_children_u18": applicant.get("num_children_u18", 0),
            "assets": applicant.get("assets", 0.0),
            "dti": applicant.get("dti", 0.0),
            "age": applicant.get("age", 0),
            "experienced_bankruptcy": applicant.get("experienced_bankruptcy", False),
            "cr_line_duration_years": applicant.get("cr_line_duration_years", 0),
        }

        ml_applicant_info = pd.DataFrame([{
            "loan_amnt": applicant.get("loan_amnt", 0.0),
            "term": applicant.get("term", ""),
            "int_rate": applicant.get("int_rate", 0.0),
            "emp_length": applicant.get("emp_length", ""),
            "home_ownership": applicant.get("home_ownership", ""),
            "annual_inc": applicant.get("annual_inc", 0.0),
            "purpose": applicant.get("purpose", ""),
            "delinq_2yrs": applicant.get("delinq_2yrs", 0),
            "open_acc": applicant.get("open_acc", 0),
            "pub_rec": applicant.get("pub_rec", 0),
            "revol_bal": applicant.get("revol_bal", 0.0),
            "repay_fail": applicant.get("repay_fail", 0)
        }]) if hardcode_ml_score is None else None

        return self.make_decision(ml_applicant_info, symbolic_applicant_info, hardcoded_ml_score=hardcode_ml_score)


if __name__ == '__main__':
    import test_applicants as test_apps
    logic = LogicComponent()

    for label, applicant, info, ml_score in test_apps.test_cases:
        logic.logger.info("\n" + "="*60)
        logic.logger.info(f"STARTING TEST CASE: {label.upper()}")
        logic.logger.info("="*60)

        decision = logic.make_decision(applicant, applicant_info=info, hardcoded_ml_score=ml_score)

        logic.logger.info(f"\nTest case '{label}' completed successfully")
        logic.logger.info("="*60)  
        
    logic.logger.info("\n" + "="*60)
    logic.logger.info("ALL TEST CASES COMPLETED")
    logic.logger.info("="*60)


