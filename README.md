# AI-Powered-Medical-Billing-and-Coding-solution-with-Fraud-Detection

Project Title:
--------------
AI-Powered Medical Billing and Coding Solution using Machine Learning and NLP with Fraud Detection

Overview:
---------
This project presents an end-to-end intelligent solution that automates the healthcare billing process by integrating AI-driven medical coding and fraud detection. It assigns ICD-10 billing codes based on a patient’s diagnosis and evaluates the legitimacy of billing claims using a machine learning classifier.

The application is built using Python and Streamlit, and it allows users to upload patient billing data in CSV format. It automatically processes the data, assigns billing codes, and predicts if a claim appears fraudulent. The results are presented in an interactive interface with an option to download them for further use.

Key Features:
-------------
- Assigns ICD-10 codes based on the Medical Condition column using rule-based NLP
- Predicts the possibility of fraudulent billing using a trained Random Forest classifier
- Interactive Streamlit interface for CSV upload, analysis, and download
- Built-in label encoders and fraud detection model
- Real-time analysis of healthcare financial data

Technologies Used:
------------------
- Python 3.x
- Streamlit
- Pandas
- Scikit-learn
- Joblib (for model persistence)

Installation & Setup:
---------------------
1. Clone this repository or download the ZIP.
2. Navigate to the project directory.

   cd your-project-directory

3. Install dependencies:

   pip install -r requirements.txt

   If requirements.txt is missing, install manually:

   pip install pandas streamlit scikit-learn joblib

4. Ensure these files are present in the same directory:
   - app.py — main Streamlit application
   - fraud_detection_model.pkl — trained ML model
   - label_encoders.pkl — label encoders used in training
   - healthcare_dataset_final_with_icd.csv — sample input dataset

5. Run the application:

   streamlit run app.py

Usage:
------
1. Upload a CSV file containing billing data. Required columns include:
   - Medical Condition
   - Age, Gender, Blood Type
   - Billing Amount
   - Insurance Provider

2. The app will:
   - Assign ICD-10 codes to each row based on the diagnosis.
   - Predict whether each billing claim is potentially fraudulent.
   - Display results in a clean, interactive table.

3. You can download the output as a CSV for records or reporting.

Expected Output:
----------------
The system will generate a table including:
- Medical Condition
- ICD-10 Code
- Billing Amount
- Fraud Prediction (Legitimate or Fraud)

Folder Structure:
-----------------
- app.py                        → Main Streamlit UI
- fraud_detection_model.pkl    → Pre-trained RandomForest model
- label_encoders.pkl           → Encoders for categorical data
- healthcare_dataset_final_with_icd.csv → Sample input dataset
- README.txt                   → Project documentation

Limitations & Future Scope:
----------------------------
- The ICD-10 assignment is rule-based and can be expanded using NLP models.
- Fraud detection is trained on a simulated condition (high billing); real-world data would improve accuracy.
- Future versions may include blockchain integration for claim tracking and security.

Contributors:
-------------
- Aryan Sharma (Developer & Researcher)

License:
--------
This project is licensed under the MIT License. Feel free to use, modify, and share with attribution.

Contact:
--------
For queries or contributions, contact Aryan Sharma on GitHub or LinkedIn.
