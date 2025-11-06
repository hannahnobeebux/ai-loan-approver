``` mermaid
classDiagram
    class Logging {
        info(message)
    }

    class Model {
        logger: Logging
        data: pd.DataFrame
        num_cols: array
        cat_cols: array
        X_train: pd.DataFrame 
        y_train: pd.DataFrame 
        X_cal: pd.DataFrame 
        y_cal: pd.DataFrame 
        X_test: pd.DataFrame 
        y_test: pd.DataFrame
        model: LogisticRegression

        load_data(filepath, num_cols, cat_cols, nrows)
        clean_data()
        process_data()
        train_model()
        test_model()
        process_application(X) float
    }

    class Decision {
        outcome: string
        ml_score: float
        symbolic_score: float
        reasons: array
    }

    class RuleScorer {
        score_ml: float
        defaults: Dict

        set_score_ml(score)
        feature1_adj(num_children) float
        feature2_adj(salary) float
        feature3_adj(assets) float
        adjust_score_and_decide(applicant_info) Decision
    }

    class LogicComponent {
        logger: Logging
        model: Model
        num_cols: array
        cat_cols: array
        filename: string
        nrows: int
        rules: RuleScorer

        use_ml_model(applicant) float
        make_decision(sample_info, applicant_info) Decision
    }

    class GUI {
        components: Dict
        tests: Tests

        create_window()
        centre_sized_window(width, height)
        add_application_page()
        submit_application()
        handle_tests()
        load_test_data(test)
    }

    class Tests {
        test_case_1: Dict,
        test_case_2: Dict,
        test_case_3: Dict
    }

    GUI --> Tests
    GUI --> LogicComponent
    LogicComponent --> Logging
    LogicComponent --> RuleScorer
    RuleScorer --> Decision
    LogicComponent --> Model
    Model --> Logging
```