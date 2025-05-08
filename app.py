import streamlit as st
import os
import pandas as pd
from PIL import Image
from inference import predict

st.set_page_config(page_title="🩺 Diabetic Retinopathy Grader", layout="wide")

# Paths
OUTPUT_DIR = "output"
TEST_RESULTS_DIR = os.path.join(OUTPUT_DIR, "Test_Results")

# Load available models
trained_models = [
    d for d in os.listdir(OUTPUT_DIR)
    if os.path.isdir(os.path.join(OUTPUT_DIR, d)) and d != "Test_Results"
]

# Sidebar – Model Selection
st.sidebar.title("📁 Model Selection")
if not trained_models:
    st.sidebar.error("❌ No trained models found in output/")
    st.stop()

selected_model = st.sidebar.selectbox("Choose a Trained Model", trained_models)
model_results_dir = os.path.join(TEST_RESULTS_DIR, selected_model)

# Sidebar – Model Info
st.sidebar.title("📊 Model Metrics")

summary_file = os.path.join(model_results_dir, "test_summary.txt")
if os.path.exists(summary_file):
    with open(summary_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        st.sidebar.write(line.strip())
else:
    st.sidebar.write("No test summary available.")

# Sidebar – Confusion Matrix
conf_matrix_path = os.path.join(model_results_dir, "confusion_matrix.png")
if os.path.exists(conf_matrix_path):
    st.sidebar.image(conf_matrix_path, caption="Confusion Matrix", use_column_width=True)

# Sidebar – Accuracy Progress Bar
predictions_csv = os.path.join(model_results_dir, "predictions.csv")
if os.path.exists(predictions_csv):
    df = pd.read_csv(predictions_csv)
    accuracy = (df['actual'] == df['predicted']).mean() * 100
    st.sidebar.metric("📈 Accuracy", f"{accuracy:.2f}%")
    st.sidebar.progress(accuracy / 100.0)

    # Top Confusions
    misclassified = df[df['actual'] != df['predicted']]
    confusions = misclassified.groupby(['actual', 'predicted']).size().reset_index(name='count')
    if not confusions.empty:
        st.sidebar.write("🔁 Most Frequent Confusions")
        st.sidebar.dataframe(confusions.sort_values("count", ascending=False).head(5))

# Main Area – Title
st.title("🩺 Diabetic Retinopathy Detection")
st.markdown("Upload a retina image to get instant diagnosis using your trained model.")

# Image Upload
uploaded_file = st.file_uploader("📤 Upload Retina Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temporary image
    tmp_path = "temp_uploaded.jpg"
    image.save(tmp_path)

    if st.button("🔍 Analyze Image"):
        result = predict(tmp_path, selected_model)

        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader(f"📣 Diagnosis: {result['severity']}")
            st.write(f"💡 Confidence: {result['confidence']}%")

            st.image(result["chart_path"], caption="Class Probabilities", use_column_width=True)
            st.image(result["conf_matrix_path"], caption="Sample Confusion Matrix", use_column_width=True)
