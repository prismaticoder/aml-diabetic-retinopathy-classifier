# Importing Streamlit for creating the web UI
import streamlit as st

# Importing standard Python libraries
import os  # for file system paths
import pandas as pd  # for reading CSV files (used to show predictions)

# Set up the layout and title of the web page
st.set_page_config(page_title="Diabetic Retinopathy Detector", layout="wide")

# Define the path where output folders (trained models and results) are saved
output_dir = "output"
test_results_dir = os.path.join(output_dir, "Test_Results")

# Get all trained model folders from the "output" directory, ignore Test_Results folder
# We only want to list actual model folders for user selection
trained_models = [
    d for d in os.listdir(output_dir)
    if os.path.isdir(os.path.join(output_dir, d)) and d != "Test_Results"
]

# If no models are found, display error in sidebar and stop the app
if not trained_models:
    st.sidebar.write("❌ No trained models found! Please train a model first.")
    st.stop()

# Create a dropdown in the sidebar to allow the user to select a trained model
selected_model = st.sidebar.selectbox("Select Model", trained_models)

# Create the full path for the selected model's test results
model_test_dir = os.path.join(test_results_dir, selected_model)

# Add a heading in the sidebar for showing model details
st.sidebar.header("📊 Model Details")

# Path of the summary file (containing test info like accuracy, samples)
summary_path = os.path.join(model_test_dir, "test_summary.txt")

# If the summary file exists, read and display important lines in sidebar
if os.path.exists(summary_path):
    with open(summary_path, "r") as f:
        summary_lines = f.readlines()
    
    # Remove lines that show file paths, keep only meaningful information
    filtered_lines = [
        line.strip() for line in summary_lines
        if "Confusion Matrix saved at:" not in line and "Predictions CSV saved at:" not in line
    ]

    # Display each line in sidebar
    for line in filtered_lines:
        st.sidebar.write(line)

# Path to the confusion matrix image
conf_matrix_path = os.path.join(model_test_dir, "confusion_matrix.png")
# If available, show the confusion matrix image in the sidebar
if os.path.exists(conf_matrix_path):
    st.sidebar.image(conf_matrix_path, caption=f"{selected_model} - Confusion Matrix")

# Path to the CSV file that contains predictions of the model
predictions_path = os.path.join(model_test_dir, "predictions.csv")

# If prediction results are available
if os.path.exists(predictions_path):
    st.sidebar.subheader("📊 Model Performance Breakdown")

    # Load the CSV containing True and Predicted Labels
    df = pd.read_csv(predictions_path)

    # Calculate accuracy = (correct predictions / total predictions)
    accuracy = (df["True Label"] == df["Predicted Label"]).mean() * 100

    # Show accuracy as a metric (with progress bar)
    st.sidebar.metric(label="📊 Test Set Accuracy", value=f"{accuracy:.2f}%")
    st.sidebar.write("📈 **Accuracy Breakdown:**")
    st.sidebar.progress(accuracy / 100)

    # Identify where the model made mistakes
    misclassifications = df[df["True Label"] != df["Predicted Label"]]

    # Group by the most common mistakes
    most_common_mistakes = misclassifications.groupby(
        ["True Label", "Predicted Label"]
    ).size().reset_index(name="Count")

    # Display top 5 most common misclassifications
    if not most_common_mistakes.empty:
        st.sidebar.write("🔄 **Most Common Misclassifications:**")
        st.sidebar.dataframe(most_common_mistakes.sort_values("Count", ascending=False).head(5))

# Main title on the webpage
st.title("🩺 Diabetic Retinopathy Detection AI")

# Allow user to upload retina images (supports jpg, png, jpeg)
uploaded_file = st.file_uploader("📤 Upload Retina Image", type=["jpg", "png", "jpeg"])

# If a file is uploaded
if uploaded_file:
    # Import libraries only when needed
    from PIL import Image
    from inference import predict  # Import your inference function

    # Open the image using PIL
    image = Image.open(uploaded_file)

    # Show the uploaded image to the user
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save the uploaded image temporarily for prediction
    temp_path = "temp.jpg"
    image.save(temp_path)

    # When user clicks on "Analyze Image" button
    if st.button("🔍 Analyze Image"):
        result = predict(temp_path, selected_model)

        # Handle error if something goes wrong
        if "error" in result:
            st.error(result["error"])
        else:
            # Show prediction and confidence
            st.subheader(f"🩺 **Diagnosis: {result['severity']}**")
            st.write(f"📊 **Confidence:** {result['confidence']}%")

            # Show chart of class probabilities
            st.image(result["chart_path"], caption="Class Probabilities", use_container_width=True)

            # Show confusion matrix (random/fake for now as placeholder)
            st.image(result["conf_matrix_path"], caption="Confusion Matrix", use_container_width=True)
