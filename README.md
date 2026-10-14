**Steps to Download and Run the Project (Python + Streamlit, Windows)**

1. Install Python

* Download and install Python (3.8 or above) from the official website
* During installation, enable **“Add Python to PATH”**

2. Install Git

* Download and install Git from https://git-scm.com
* Verify installation using:
  `git --version`

3. Clone the GitHub Repository

* Open Command Prompt / PowerShell
* Run:
  `https://github.com/ItsSuriya/Signlens--Sign-langauge-Interpreter/projects?query=is%3Aopen`
* Move into the project folder:
  `cd <repository-name>`

4. Create Virtual Environment (Recommended)

* Run:
  `python -m venv venv`
* Activate it:
  `venv\Scripts\activate`

5. Install Requirements

* Place your `requirements.txt` file in the project folder
* Run:
  `pip install -r requirements.txt`

6. Run the Streamlit Application

* Identify the main file ('app.py`)
* Run:
  `streamlit run app.py`

7. Open in Browser

* Streamlit will automatically open in browser
* If not, go to:
  `http://localhost:8501`
