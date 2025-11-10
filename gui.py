import tkinter as tk
from tkinter import messagebox

from gui_tests import Tests
from logic import LogicComponent

# A class used to represent the frontend of the application
class Gui:

    def __init__(self):
        # Components to contain the GUI instances
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
        # Create a centered window
        self.window.title("AI Loan Manager - Prototype")
        self.centre_sized_window(800, 500)
        self.add_widgets()

    def centre_sized_window(self, width, height):
        self.window.update_idletasks()
        # Centre the window with given dimensions
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.window.geometry(f"{width}x{height}+{x}+{y}")

    def add_widgets(self):
        # Add the application page (future proof for more pages)
        self.add_application_page()

    def add_application_page(self):
        # Add title
        title = tk.Label(text='Loan Application Page')
        title.pack()

        # Create a frame for the form
        frame = tk.Frame(master=self.window,borderwidth=2)

        # Number of entires per row in the form
        columns = 4
        counter = 0
        # Produce an entry witha  label for all components
        for key in self.components:
            # Frame for label + entry
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

        # Add submit button for the form
        submit = tk.Button(master=self.window, text="Submit Application", command=self.submit_application)
        submit.pack()

    def submit_application(self):
        application_values = {}
        # Define string entreis
        string_vals = ["home_ownership", "purpose", "employment_type"]
        for key, value in self.components.items():
            # Obtain entry value
            component_val = value.get()
            # Handle missing entries
            if not component_val:
                messagebox.showerror("Validation Error", f"Value '{key}' must be supplied.")
                return 0
            if key in string_vals:
                application_values[key] = component_val
            else:
                # Handle non-string entries (if not string, assume float)
                try:
                    application_values[key] = float(component_val)
                except ValueError:
                    print(component_val)
                    messagebox.showerror("Validation Error", f"Value '{key}' must be numeric.")
                    return 0
        print(application_values)
        # Call and obtain decision from the Logic component
        decision = self.logic.process_application(application_values)
        response = decision.outcome
        ml_score = round(decision.ml_score, 0)
        reasons = decision.reasons
        symbol_score = decision.symbolic_score
        output = "Approved" if response == "APPROVE" else "Denied"
        # Output the decision
        messagebox.showinfo("Loan App Outcome", f"Loan {output}!\n \
                            ML Score: {ml_score}\n \
                            Symbolic Score: {symbol_score}\n \
                            Reasons: {reasons}")

    def handle_tests(self):
        # Create test buttons to load applications quickly
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
        # Load test values into the entries depending on the test button pressed
        for key, value in self.components.items():
            value.delete(0, tk.END)
            value.insert(0, test[key])

if __name__ == '__main__':
    # Create the GUI
    frontend = Gui()
    # Create the test buttons
    frontend.handle_tests()
    # Start the GUI application
    frontend.window.mainloop()