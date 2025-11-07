# AI Loan Approver (Prototype) - README File

This README file should be used as instructions to install and run the application.

## Setting Up the Development Environment

To ensure consistency and manage dependencies efficiently, we recommend using a virtual environment for local development. Follow the steps below to set up your development environment.

### Prerequisites

Ensure you have **Python 3.13** or higher and **pip** installed on your machine:

```
python --version
pip --version
```

### Creating a Virtual Environment
Navigate to the root of the project and create a virtual environment:
```
# Create a virtual environment named 'venv'
python -m venv venv
```

### Activating the Virtual Environment
Before installing the dependencies, you need to activate the virtual environment:

- *Windows*
```
.\venv\Scripts\activate
```

- *Linux*
```
source venv/Scripts/activate
```
You should now see (venv) prefixed to your terminal, indicating that the virtual environment is active. You can then select this as the chosen Python Interpreter within your IDE.

### Confirm Virtual Environment:
In order to confirm the right Python interpreter is being used from the virtual environment, you can check the path of the Python executable:
- *Windows*
```
where python
```
- *Linux*
```
which python
```
The output should point to the Python interpreter within the venv directory of your project.

### Installing Dependencies

With the virtual environment activated, install the required packages using the **requirements.txt** file:
```
pip install -r requirements.txt
```
This command will install all the necessary Python packages listed in **requirements.txt** within your virtual environment.

### Deactivating the Virtual Environment
After you've completed your work or if you need to exit the virtual environment, you can deactivate it by running:
```
deactivate
```

This command will return you to the global Python environment.


## Running The Application

While many of the python files can be executed and run, for the full application, run the **gui.py** file using your python command (python, python3).

```
python gui.py
```