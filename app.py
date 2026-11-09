import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_FILE = Path("wine_model.pkl")
DATA_FILE = Path("wine_quality.csv")

st.set_page_config(page_title="Wine Quality Prediction System", page_icon="🍷", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

st.title("🍷 Wine Quality Prediction System")
st.caption("Classroom prototype for preliminary red-wine quality classification.")

if not MODEL_FILE.exists():
    st.error("Model file not found. Run `python train_model.py` first.")
    st.stop()
if not DATA_FILE.exists():
    st.error("Dataset file not found. Run `python train_model.py` first.")
    st.stop()

model = load_model()
data = pd.read_csv(DATA_FILE)

c1, c2, c3 = st.columns(3)
c1.metric("Records", len(data))
c2.metric("Average quality score", round(data["quality"].mean(), 2))
c3.metric("Good-quality records", int((data["quality"] >= 6).sum()))

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

st.subheader("Predict uploaded wine samples")
st.write("Upload a CSV with the same predictor columns used during training. Do not include the quality target.")
feature_names = list(model.feature_names_in_)
template = data[feature_names].head(5)
st.download_button(
    "Download sample input CSV",
    template.to_csv(index=False),
    file_name="sample_wine_input.csv",
    mime="text/csv",
)

uploaded = st.file_uploader("Choose wine sample CSV", type=["csv"])
if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
        missing = [column for column in feature_names if column not in new_data.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            predictions = model.predict(new_data[feature_names])
            results = new_data.copy()
            results["Prediction"] = [
                "GOOD QUALITY" if value == 1 else "NEEDS IMPROVEMENT"
                for value in predictions
            ]
            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)
            st.metric("Predicted good-quality samples", int((predictions == 1).sum()))
            st.download_button(
                "Download prediction results",
                results.to_csv(index=False),
                file_name="wine_quality_predictions.csv",
                mime="text/csv",
            )
    except Exception as exc:
        st.error(f"Could not read or predict this file: {exc}")

st.info(
    "Responsible use: This is a classroom screening prototype. "
    "It is not a substitute for laboratory testing, sensory evaluation, "
    "food-safety procedures, or professional quality-control decisions."
)
