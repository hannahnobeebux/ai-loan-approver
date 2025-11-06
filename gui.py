import tkinter as tk
from tkinter import messagebox

from gui_tests import Tests
from logic import LogicComponent

class Gui:

    def __init__(self):
        self.components = {
            "loan_amnt": None,
            "term": None,
            "int_rate": None,
            "emp_length": None,
            "home_ownership": None,
            "annual_inc": None,
            "purpose": None,
            "delinq_2yrs": None,
            "open_acc": None,
            "pub_rec": None,
            "revol_bal": None,
            "employment_type": None,
            "num_children_u18": None,
            "assets": None,
            "dti": None,
            "age": None,
            "experienced_bankruptcy": None,
            "cr_line_duration_years": None
        }
        self.tests = Tests()
        self.window = tk.Tk()
        self.logic = LogicComponent()
        self.create_window()

    def create_window(self):
        self.window.title("AI Loan Manager - Prototype")
        self.centre_sized_window(800, 500)
        self.add_widgets()

    def centre_sized_window(self, width, height):
        self.window.update_idletasks()
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.window.geometry(f"{width}x{height}+{x}+{y}")

    def add_widgets(self):
        self.add_application_page()

    def add_application_page(self):
        title = tk.Label(text='Loan Application Page')
        title.pack()

        frame = tk.Frame(master=self.window,borderwidth=2)

        columns = 4
        counter = 0
        for key in self.components:
            f = tk.Frame(master=frame)
            row = counter//columns
            col = counter%columns
            f.grid(row=row, column=col)
            label = tk.Label(text=key, master=f)
            self.components[key] = tk.Entry(master=f)
            label.pack()
            self.components[key].pack()
            counter += 1
        
        frame.pack()

        submit = tk.Button(master=self.window, text="Submit Application", command=self.submit_application)
        submit.pack()

    def submit_application(self):
        application_values = {}
        string_vals = ["home_ownership", "purpose", "employment_type"]
        for key, value in self.components.items():
            component_val = value.get()
            if not component_val:
                messagebox.showerror("Validation Error", f"Value '{key}' must be supplied.")
                return 0
            if key in string_vals:
                application_values[key] = component_val
            else:
                try:
                    application_values[key] = float(component_val)
                except ValueError:
                    print(component_val)
                    messagebox.showerror("Validation Error", f"Value '{key}' must be numeric.")
                    return 0
        print(application_values)
        decision = self.logic.process_application(application_values)
        response = decision.outcome
        ml_score = round(decision.ml_score, 0)
        reasons = decision.reasons
        symbol_score = decision.symbolic_score
        output = "Approved" if response == "APPROVE" else "Denied"
        messagebox.showinfo("Loan App Outcome", f"Loan {output}!\nML Score: {ml_score}\nSymbolic Score: {symbol_score}\nReasons: {reasons}")

    def handle_tests(self):
        tests = [
            self.tests.test_case_1,
            self.tests.test_case_2,
            self.tests.test_case_3,
            self.tests.test_case_4,
            self.tests.test_case_5,
            self.tests.test_case_6
        ]
        frame = tk.Frame(master=self.window)
        columns = 8
        for i, test in enumerate(tests):
            button = tk.Button(
                master=frame, 
                text=f"Load test {i + 1}", 
                command=lambda test=test: self.load_test_data(test)
            )
            button.grid(row=i//columns, column=i%columns)
        frame.pack()

    def load_test_data(self, test):
        for key, value in self.components.items():
            value.delete(0, tk.END)
            value.insert(0, test[key])

if __name__ == '__main__':
    frontend = Gui()
    frontend.handle_tests()
    frontend.window.mainloop()