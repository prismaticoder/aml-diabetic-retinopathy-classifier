# 📦 Import necessary libraries
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from model import get_model  # This function loads the model architecture

# ✅ Choose whether to use GPU (if available) or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔖 Define the class labels for diabetic retinopathy severity
severity_labels = {
    0: "No DR (Healthy)",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR (Most Severe)"
}

def get_recommendation(predicted_class):
    df = pd.read_csv("llm_recommendations.csv")
    return df[df["dr_grade"] == predicted_class].sample(n=1)["recommendation"].values[0]

# 🧠 Load the most recent trained model (ResNet or EfficientNet) by name
def load_latest_model(model_name):
    try:
        # 👇 Extract base model name from something like "resnet50_lr0.0001_bs32"
        base_model_name = model_name.split("_")[0]

        # 🔍 Look for a directory in 'output/' folder that starts with our model name
        available_models = [
            d for d in os.listdir("output")
            if d.lower().startswith(model_name.lower())
        ]

        if not available_models:
            raise FileNotFoundError(f"❌ No trained model found for '{model_name}'!")

        # Pick the matched directory
        model_dir = os.path.join("output", available_models[0])

        # Find the most recently saved model weight file (.pth)
        model_files = sorted(
            [f for f in os.listdir(model_dir) if f.endswith(".pth")],
            key=lambda x: os.path.getmtime(os.path.join(model_dir, x))
        )

        if not model_files:
            raise FileNotFoundError("❌ No .pth model files found!")

        # 📌 Path to the latest weight file
        model_path = os.path.join(model_dir, model_files[-1])

        # 🧠 Load the model architecture and weights
        model = get_model(base_model_name.lower()).to(device)
        checkpoint = torch.load(model_path, map_location=device)

        # 🧽 Remove unnecessary wrappers (like 'state_dict' or 'module.')
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

        try:
            model.load_state_dict(checkpoint)
        except RuntimeError:
            print("⚠️ Warning: Weight mismatch, loading with strict=False")
            model.load_state_dict(checkpoint, strict=False)

        # 🧊 Freeze BatchNorm behavior during inference
        model.eval()
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.track_running_stats = False
                module.eval()

        print(f"✅ Loaded model: {model_path}")
        return model

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise e


# 🧹 Function to prepare a retina image for prediction
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Make image size compatible with model input
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard normalization
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")  # Open image and convert to RGB
    image = transform(image).unsqueeze(0)  # Add batch dimension [1, C, H, W]
    return image.to(device)


# 🔍 Function to run the model and return prediction info
def predict(image_path, model_name):
    try:
        model = load_latest_model(model_name)
    except FileNotFoundError as e:
        return {"error": str(e)}

    image = preprocess_image(image_path)

    # 🚫 Disable gradient tracking (no training is done during prediction)
    with torch.no_grad():
        output = model(image)  # Forward pass through the model
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item() * 100  # Confidence in %
        
    recommendation = get_recommendation(predicted_class)

    # 📊 Plot class probability bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(severity_labels.values()), y=probabilities.cpu().numpy(), ax=ax)
    ax.set_title(f"Predicted: {severity_labels[predicted_class]}")
    ax.set_ylabel("Confidence (%)")
    ax.set_xlabel("Severity Class")
    ax.set_xticklabels(severity_labels.values(), rotation=20)
    plt.tight_layout()
    plt_path = "probabilities_chart.png"
    plt.savefig(plt_path)
    plt.close()

    # 📘 Show dummy confusion matrix (this is for UI visualization only)
    confusion_matrix = np.random.randint(10, 100, size=(5, 5))
    df_cm = pd.DataFrame(confusion_matrix, index=severity_labels.values(), columns=severity_labels.values())
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Example)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    conf_matrix_path = "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(conf_matrix_path)
    plt.close()

    # 📦 Return results
    return {
        "severity": severity_labels[predicted_class],
        "confidence": round(confidence, 2),
        "probabilities": {severity_labels[i]: round(prob.item() * 100, 2) for i, prob in enumerate(probabilities)},
        "chart_path": plt_path,
        "conf_matrix_path": conf_matrix_path,
        "recommendation": recommendation
    }
