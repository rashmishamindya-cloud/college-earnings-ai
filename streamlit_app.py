import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(
    page_title="College Earnings AI",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load("college_earnings_model.pkl")
    with open("feature_metadata.json", "r") as f:
        metadata = json.load(f)
    return model, metadata

try:
    model, metadata = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

st.title("🎓 College Earnings Prediction AI")
st.write("Predict the earnings trajectory category of a college programme.")

feature_names = metadata["feature_names"]
numeric_features = metadata["numeric_features"]
categorical_features = metadata["categorical_features"]
defaults = metadata["defaults"]

with st.form("prediction_form"):
    st.subheader("Enter Programme Details")

    values = {}

    for feature in feature_names:
        default = defaults.get(feature)

        if feature in numeric_features:
            values[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=float(default if default is not None else 0.0)
            )
        else:
            values[feature] = st.text_input(
                feature.replace("_", " ").title(),
                value=str(default if default is not None else "")
            )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([values], columns=feature_names)

    try:
        prediction = model.predict(input_df)[0]

        st.success(f"Predicted Earnings Category: {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = model.classes_

            st.subheader("Prediction Probabilities")

            results = pd.DataFrame({
                "Category": classes,
                "Probability (%)": probabilities * 100
            })

            st.bar_chart(
                results.set_index("Category")
            )

    except Exception as e:
        st.error(f"Prediction error: {e}")