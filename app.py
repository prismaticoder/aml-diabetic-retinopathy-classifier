import streamlit as st
import os
import pandas as pd
from PIL import Image
from inference import predict

st.set_page_config(page_title="Diabetic Retinopathy Detector", layout="wide")

# Paths
output_dir = "output"
test_results_dir = os.path.join(output_dir, "Test_Results")

# Fetch trained models
trained_models = [
    d for d in os.listdir(output_dir)
    if os.path.isdir(os.path.join(output_dir, d)) and d != "Test_Results"
]

if not trained_models:
    st.sidebar.error("❌ No trained models found! Please train a model first.")
    st.stop()

# Sidebar: Model selection
selected_model = st.sidebar.selectbox("Select Trained Model", trained_models)
model_test_dir = os.path.join(test_results_dir, selected_model)

# Sidebar: Model Info
st.sidebar.header("📊 Model Information")

summary_path = os.path.join(model_test_dir, "test_summary.txt")
if os.path.exists(summary_path):
    with open(summary_path, "r") as f:
        for line in f:
            if "Precision" in line:
                st.sidebar.subheader("📉 Evaluation:")
            st.sidebar.write(line.strip())

conf_matrix_path = os.path.join(model_test_dir, "confusion_matrix.png")
if os.path.exists(conf_matrix_path):
    st.sidebar.image(conf_matrix_path, caption="Confusion Matrix")

predictions_path = os.path.join(model_test_dir, "predictions.csv")
if os.path.exists(predictions_path):
    st.sidebar.subheader("🧪 Performance Breakdown")
    df = pd.read_csv(predictions_path)
    accuracy = (df["actual"] == df["predicted"]).mean() * 100
    st.sidebar.metric("✅ Accuracy", f"{accuracy:.2f}%")
    st.sidebar.progress(accuracy / 100)

    misclassifications = df[df["actual"] != df["predicted"]]
    top_mistakes = (
        misclassifications.groupby(["actual", "predicted"]).size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
        .head(5)
    )

    if not top_mistakes.empty:
        st.sidebar.write("🔁 Frequent Misclassifications")
        st.sidebar.dataframe(
            top_mistakes.rename(columns={"actual": "True", "predicted": "Predicted"})
        )

# Main area
st.title("🩺 Diabetic Retinopathy Detection")

uploaded_file = st.file_uploader("📤 Upload Fundus Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Retina Image", use_container_width=True)

    image_path = "temp_infer_image.jpg"
    image.save(image_path)

    if st.button("🔍 Run Diagnosis"):
        result = predict(image_path, selected_model)

        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader(f"🧠 Diagnosis: {result['severity']} ({result['confidence']}%)")
            st.image(result["chart_path"], caption="📊 Class Probabilities", use_column_width=True)
            st.image(result["conf_matrix_path"], caption="🧮 Example Confusion Matrix", use_column_width=True)
            st.write("### All Probabilities")
            st.json(result["probabilities"])
