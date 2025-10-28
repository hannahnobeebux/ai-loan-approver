from typing import Any, Dict, Optional
from model import Model


class Decision:
    outcome: str   #approve or deny
    ml_score: int     #credit risk score from the model
    symbolic_score: int  #credit risk score after applying rules

# baseline credit score: 900
# each penalty point reduces score by 10 points
# each child beyond 3 reduces score by configured penalty


class RuleScorer:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        
        defaults = {
            # Decision thresholds on final score
            # in between deny_threshold and approve_threshold is "manual review"
            # eg: 900 is 10% default risk
            "deny_threshold": 700,
            "approve_threshold": 900,

            # salary (avg annual income over 5 years)
            # increase points when there's a bonus
            # 6pts for each 10k but cap at 60pts
            "salary_bonus_per_10k": 6.0,     
            "salary_bonus_cap": 60.0,       

            # employment - penalty if self employed, bonus if full-time employed
            "employment_penalty_self_employed": -40.0,
            "employment_bonus_fte": +10.0,            

            # ask for number of children under 18
            # Children penalties - the higher the number of children, the higher the penalty
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

    def penalise_children_u18(self, num_children: int) -> int:
        # Penalise applicants with more than 3 children by reducing their credit score.
        # consider ages of children
        if num_children is None or np.isnan(num_children):
            return 0.0
        n = int(num_children)
        if n >= 4:
            return self.cfg["children_penalty_ge4"]
        return float(self.cfg["children_penalties"].get(n, 0.0))

    def salary_points(self, salary: float) -> float:
        if salary is None or np.isnan(salary):
            return 0.0

    def assets_points() :
        pass

    def dti_penalty() :
        pass

    def salary_and_assets() :
        # call salary_points and assets_points
        # no need to call reduce credit risk with outcome of salary and assets individual scores



        pass

# amend credit risk after decision rules

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

        self.model.load_data("loan_data.csv", numeric, categorical, 2000)
        self.model.train_model()
        self.model.test_model()
        
    
    def decide(self, score: int) -> str:
        # make a decision based on the final credit score.
        if score < self.rules["deny_threshold"]:
            return "deny"
        elif score >= self.rules["approve_threshold"]:
            return "approve"
        else:
            return "manual review"


if __name__ == '__main__':
    logic = LogicComponent()

    # pass in sample application - coming from GUI in the form of a JSON
    # pass in extra info such as number of children, assets, employment type, etc.

    # using logic.decide() to return approve/deny
    # if in betweeen threshholds - in review - extra function which checks if salary is high enough etc..
    # mvp = in between thresholds - deny 
