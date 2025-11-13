import streamlit as st
import pandas as pd
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🎓 Student Dropout Prediction (Batch Mode)",
    page_icon="📊",
    layout="wide"
)

st.title("🎓 Student Dropout Prediction (File Upload Mode)")
st.markdown("""
Upload a **CSV or Excel file** containing student records with **integer/float values only**.  
The system will predict whether each student is likely to **Graduate**, **Dropout**, or remain **Enrolled**.
""")

# --- 2. LOAD MODEL ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    try:
        # Load model and encoder together
        model_and_encoder = joblib.load('model_and_encoder.joblib')
        model = model_and_encoder['model']
        encoder = model_and_encoder['encoder']
        # Load model columns
        model_columns = joblib.load('model_columns.joblib')
        return model, encoder, model_columns
    except FileNotFoundError:
        st.error("🚨 Missing model files. Ensure 'model_and_encoder.joblib' and 'model_columns.joblib' are in this directory.")
        return None, None, None

model, encoder, model_columns = load_artifacts()

# --- 3. HELPER FUNCTION ---
def predict_from_file(df):
    """Preprocess uploaded data and return predictions and probabilities."""
    if model is None or encoder is None or model_columns is None:
        return None, None

    # Ensure only numeric data
    df = df.select_dtypes(include=['int64', 'float64'])

    # Align columns with model training features
    aligned_df = pd.DataFrame(columns=model_columns, dtype=float)
    df_processed = pd.get_dummies(df, drop_first=True)
    aligned_df = pd.concat([aligned_df, df_processed], axis=0, sort=False).fillna(0)
    aligned_df = aligned_df[model_columns]

    # Make predictions
    preds_encoded = model.predict(aligned_df)
    probs = model.predict_proba(aligned_df)
    preds_labels = encoder.inverse_transform(preds_encoded)

    return preds_labels, probs

# --- 4. FILE UPLOAD SECTION ---
uploaded_file = st.file_uploader("📂 Upload your student data file (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read file depending on extension
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("✅ File successfully uploaded and read!")
        st.dataframe(df.head())

        # --- 5. PREDICTION ---
        preds, probs = predict_from_file(df)

        if preds is not None:
            st.subheader("📊 Prediction Results")

            # Convert probabilities to dataframe
            prob_df = pd.DataFrame(probs, columns=[f"Prob_{cls}" for cls in encoder.classes_])
            results = pd.concat([df.reset_index(drop=True), pd.Series(preds, name='Predicted_Outcome'), prob_df], axis=1)

            st.dataframe(results)

            # --- 6. DOWNLOAD OPTION ---
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv,
                file_name="student_predictions.csv",
                mime="text/csv"
            )
        else:
            st.error("Prediction failed — please check that your file has numeric data only.")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

else:
    st.info("📥 Please upload a CSV or Excel file to begin.")
