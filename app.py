import streamlit as st
import os
import pandas as pd
from PIL import Image
from inference import predict

st.set_page_config(page_title="Diabetic Retinopathy Detector", layout="wide")

output_dir = "output"
test_results_dir = os.path.join(output_dir, "Test_Results")

# 🔍 Detect available models (excluding Test_Results)
trained_models = [
    d for d in os.listdir(output_dir)
    if os.path.isdir(os.path.join(output_dir, d)) and d != "Test_Results"
]

if not trained_models:
    st.sidebar.warning("⚠️ No trained models found. Please run training first.")
    st.stop()

selected_model = st.sidebar.selectbox("📦 Select Trained Model", sorted(trained_models, reverse=True))

# ℹ️ Load model summary metrics
summary_path = os.path.join(test_results_dir, selected_model, "test_summary.txt")
if os.path.exists(summary_path):
    st.sidebar.subheader("📋 Evaluation Summary")
    with open(summary_path) as f:
        for line in f.readlines():
            st.sidebar.write(line.strip())

# 📊 Display confusion matrix image
conf_matrix_path = os.path.join(test_results_dir, selected_model, "confusion_matrix.png")
if os.path.exists(conf_matrix_path):
    st.sidebar.image(conf_matrix_path, caption="Confusion Matrix", use_column_width=True)

# 📈 Load test predictions CSV
predictions_path = os.path.join(test_results_dir, selected_model, "predictions.csv")
if os.path.exists(predictions_path):
    df = pd.read_csv(predictions_path)
    accuracy = (df["actual"] == df["predicted"]).mean() * 100

    st.sidebar.metric(label="✅ Test Accuracy", value=f"{accuracy:.2f}%")
    st.sidebar.progress(accuracy / 100)

    # 🧪 Misclassifications
    misclassified = df[df["actual"] != df["predicted"]]
    if not misclassified.empty:
        summary = misclassified.groupby(["actual", "predicted"]).size().reset_index(name="Count")
        st.sidebar.subheader("🔁 Top Misclassifications")
        st.sidebar.dataframe(summary.sort_values("Count", ascending=False).head(5))

st.title("🩺 Diabetic Retinopathy Prediction")

uploaded_file = st.file_uploader("📤 Upload a Retina Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Retina Image", use_column_width=True)

    temp_path = "temp_uploaded_image.jpg"
    image.save(temp_path)

    if st.button("🔍 Run Diagnosis"):
        result = predict(temp_path, selected_model)

        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader(f"🧠 Predicted Severity: **{result['severity']}**")
            st.write(f"📊 Confidence: **{result['confidence']}%**")
            st.image(result["chart_path"], caption="Class Probabilities", use_column_width=True)
            st.image(result["conf_matrix_path"], caption="Confusion Matrix", use_column_width=True)
