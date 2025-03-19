import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from model import get_model
import numpy as np
import pandas as pd
# Load model dynamically
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define severity labels
severity_labels = {
    0: "No DR (Healthy)",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR (Most Severe)"
}


import os
import torch
from model import get_model

# Function to load a specific trained model based on user selection
def load_latest_model(model_name):
    try:
        # Ensure correct format (e.g., 'ResNet50_lr0.0001_bs32')
        model_name_parts = model_name.split("_")

        # Extract base model name
        base_model_name = model_name_parts[0]  
        
        # Construct model directory path
        # model_dir = os.path.join("output", model_name) 
        # model_dir = os.path.join("output", model_name.lower()) 
        # Ensure model selection is case-insensitive and accurate
        available_models = [
            d for d in os.listdir("output")
            if d.lower().startswith(model_name.lower())  # Ensure correct model selection
        ]

        if not available_models:
            raise FileNotFoundError(f"❌ No trained model found for '{model_name}'! Please train the model first.")

        # Use the exact directory name for the selected model
        model_dir = os.path.join("output", available_models[0]) 
 

        # Ensure the directory exists
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"❌ No trained model directory found for '{model_name}'!")

        # List model checkpoint files (sorted by modification time)
        model_files = sorted(
            [f for f in os.listdir(model_dir) if f.endswith(".pth")], 
            key=lambda x: os.path.getmtime(os.path.join(model_dir, x))
        )

        if not model_files:
            raise FileNotFoundError(f"❌ No trained model weights found in '{model_dir}'! Check if training completed.")

        # Select the latest checkpoint for the specific model
        model_path = os.path.join(model_dir, model_files[-1])

        # Load model dynamically
        model = get_model(base_model_name.lower()).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Some checkpoints save inside "state_dict", extract if necessary
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        # Remove 'module.' prefix if saved from a DataParallel model
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

        # Load state dict with strict=False to allow minor mismatches
        # model.load_state_dict(checkpoint, strict=False)

        try:
            model.load_state_dict(checkpoint)
        except RuntimeError as e:
            print(f"❌ Warning: Model weight mismatch. Loading with strict=False\n{e}")
            model.load_state_dict(checkpoint, strict=False)


        # model.eval()
        model.eval()
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.track_running_stats = False
                module.eval()


        print(f"✅ Loaded Model: {model_path}")
        return model

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise e





# Define preprocessing function
def preprocess_image(image_path):
    """Preprocesses the input image for model inference."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Prediction function with better visualization
def predict(image_path, model_name):
    try:
        model = load_latest_model(model_name)
    except FileNotFoundError as e:
        return {"error": str(e)}

    image = preprocess_image(image_path)
    
    # Get prediction and probabilities
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]  # Convert to probability distribution
        predicted_class = torch.argmax(output, 1).item()
        confidence = probabilities[predicted_class].item() * 100  # Convert to percentage

    # **Plot Class Probabilities**
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(severity_labels.values()), y=probabilities.cpu().numpy(), ax=ax)
    ax.set_title(f"Class Probabilities for {severity_labels[predicted_class]}")
    ax.set_ylabel("Probability (%)")
    ax.set_xlabel("Class")
    ax.set_xticklabels(severity_labels.values(), rotation=20)
    plt.tight_layout()  # Ensures the labels don’t get cropped

    plt_path = "probabilities_chart.png"
    plt.savefig(plt_path)
    plt.close()

    # **Generate Fake Confusion Matrix (Example)**
    confusion_matrix = np.random.randint(10, 100, size=(5, 5))  # Placeholder for real confusion matrix
    df_cm = pd.DataFrame(confusion_matrix, index=severity_labels.values(), columns=severity_labels.values())

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Example)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    # Rotate labels for better visibility
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)  # Keep y-axis labels straight
    conf_matrix_path = "confusion_matrix.png"
    plt.tight_layout()  # Ensures the labels don’t get cropped

    plt.savefig(conf_matrix_path)
    plt.close()

    return {
        "severity": severity_labels[predicted_class],
        "confidence": round(confidence, 2),
        "probabilities": {severity_labels[i]: round(prob.item() * 100, 2) for i, prob in enumerate(probabilities)},
        "chart_path": plt_path,
        "conf_matrix_path": conf_matrix_path  # Include confusion matrix
    }

# # Test Example
# if __name__ == "__main__":
#     model_name = "resnet50_lr0.0001_bs32"  # Example model name
#     test_image = "test_image.jpeg"  # Replace with an actual image path
#     result = predict(test_image, model_name)

#     if "error" in result:
#         print(result["error"])
#     else:
#         print(f"Predicted Severity: {result['severity']} (Confidence: {result['confidence']}%)")
#         print("All probabilities:", json.dumps(result["probabilities"], indent=4))
