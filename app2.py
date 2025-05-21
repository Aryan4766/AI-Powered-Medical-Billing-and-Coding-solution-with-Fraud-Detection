import streamlit as st
import pandas as pd
import joblib

model = joblib.load("fraud_detection_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

medical_code_map = {
    "Type 2 Diabetes": "E11.9",
    "Hypertension": "I10",
    "Asthma": "J45.909",
    "Chest Pain": "R07.9",
    "Migraine": "G43.909",
    "COVID-19": "U07.1",
    "Anemia": "D64.9",
    "Pneumonia": "J18.9",
    "Viral Fever": "A91",
    "Gastric Ulcer": "K25.9"
}

def assign_medical_code(condition):
    if not isinstance(condition, str):
        return "Unknown"
    for keyword in medical_code_map:
        if keyword.lower() in condition.lower():
            return medical_code_map[keyword]
    return "Unknown"

st.set_page_config(page_title="AI Medical Billing & Coding", layout="centered")
st.title("🧾 AI-Powered Medical Billing & Coding + Fraud Detection System")

uploaded_file = st.file_uploader("Upload medical billing CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Uploaded Data")
    st.dataframe(df.head())

    diagnosis_col = None
    for possible_col in ["Medical Condition", "Diagnosis", "Condition"]:
        if possible_col in df.columns:
            diagnosis_col = possible_col
            break

    if diagnosis_col:
        df["ICD-10 Code"] = df[diagnosis_col].apply(assign_medical_code)
        st.success(f"✅ ICD-10 codes assigned based on '{diagnosis_col}' column.")
    else:
        st.warning("⚠️ No column like 'Medical Condition' found for ICD-10 coding.")

    drop_cols = ["Name", "Date of Admission", "Discharge Date", "Room Number", "Doctor", "Hospital"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    for col in df.select_dtypes(include='object').columns:
        if col in ["Medical Condition", "ICD-10 Code", "Prediction", "Fraud Prediction"]:
            continue
        if col in label_encoders:
            le = label_encoders[col]
            try:
                df[col] = le.transform(df[col].astype(str))
            except Exception as e:
                st.warning(f"⚠️ Could not encode '{col}': {e}")
                df[col] = 0
        else:
            st.warning(f"⚠️ Column '{col}' not found in encoder. Filled with zeros.")
            df[col] = 0

    df_encoded = df.select_dtypes(include='number')
    for col in model.feature_names_in_:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[model.feature_names_in_]

    if st.button("🔍 Predict Fraudulent Claims"):
        predictions = model.predict(df_encoded)
        df["Fraud Prediction"] = ["⚠️ Fraud" if pred else "✅ Legitimate" for pred in predictions]
        st.subheader("📊 Prediction Results")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results CSV", data=csv, file_name="prediction_results.csv", mime='text/csv')
else:
    st.info("📂 Upload a CSV file to begin.")