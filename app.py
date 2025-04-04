import streamlit as st
import os
import pandas as pd

st.set_page_config(page_title="Diabetic Retinopathy Detector", layout="wide")

# Define paths
output_dir = "output"
test_results_dir = os.path.join(output_dir, "Test_Results")

# Get available trained models (excluding Test_Results)
trained_models = [
    d for d in os.listdir(output_dir) 
    if os.path.isdir(os.path.join(output_dir, d)) and d != "Test_Results"
]

if not trained_models:
    st.sidebar.write("❌ No trained models found! Please train a model first.")
    st.stop()

# Model selection
if trained_models:
    selected_model = st.sidebar.selectbox("Select Model", trained_models)
else:
    st.error("❌ No trained models found in the output directory.")
    st.stop()


# Define model's test result path
model_test_dir = os.path.join(test_results_dir, selected_model)

# Display Model Information
st.sidebar.header("📊 Model Details")

# Load test summary details
summary_path = os.path.join(model_test_dir, "test_summary.txt")

if os.path.exists(summary_path):
    with open(summary_path, "r") as f:
        summary_lines = f.readlines()
    
    # Filter out lines that mention paths to confusion matrix & predictions CSV
    filtered_lines = [
        line.strip() for line in summary_lines 
        if "Confusion Matrix saved at:" not in line and "Predictions CSV saved at:" not in line
    ]

    # Display filtered information
    for line in filtered_lines:
        st.sidebar.write(line)

# Load Confusion Matrix if available
conf_matrix_path = os.path.join(model_test_dir, "confusion_matrix.png")
if os.path.exists(conf_matrix_path):
    st.sidebar.image(conf_matrix_path, caption=f"{selected_model} - Confusion Matrix")

# Load Predictions Log if available
predictions_path = os.path.join(model_test_dir, "predictions.csv")

if os.path.exists(predictions_path):
    st.sidebar.subheader("📊 Model Performance Breakdown")

    df = pd.read_csv(predictions_path)
    
    # Compute overall accuracy
    accuracy = (df["True Label"] == df["Predicted Label"]).mean() * 100


    # KPI Metric Display
    st.sidebar.metric(label="📊 Test Set Accuracy", value=f"{accuracy:.2f}%")

    # Progress Bar for Accuracy
    st.sidebar.write("📈 **Accuracy Breakdown:**")
    st.sidebar.progress(accuracy / 100)

    # Compute misclassification summary
    misclassifications = df[df["True Label"] != df["Predicted Label"]]
    most_common_mistakes = misclassifications.groupby(["True Label", "Predicted Label"]).size().reset_index(name="Count")

    # Show top misclassifications
    if not most_common_mistakes.empty:
        st.sidebar.write("🔄 **Most Common Misclassifications:**")
        st.sidebar.dataframe(most_common_mistakes.sort_values("Count", ascending=False).head(5))

st.title("🩺 Diabetic Retinopathy Detection AI")

uploaded_file = st.file_uploader("📤 Upload Retina Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    from PIL import Image
    from inference import predict

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    temp_path = "temp.jpg"
    image.save(temp_path)

    if st.button("🔍 Analyze Image"):
       result = predict(temp_path, selected_model)  # Make sure result is inside the button block

       if "error" in result:
        st.error(result["error"])
       else:
        st.subheader(f"🩺 **Diagnosis: {result['severity']}**")
        st.write(f"📊 **Confidence:** {result['confidence']}%")
        
        # Show Class Probability Chart
        st.image(result["chart_path"], caption="Class Probabilities", use_container_width=True)

        # Show Confusion Matrix
        st.image(result["conf_matrix_path"], caption="Confusion Matrix", use_container_width=True)



