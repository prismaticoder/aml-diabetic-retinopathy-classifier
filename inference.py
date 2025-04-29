import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Labels for DR stages
severity_labels = {
    0: "No DR (Healthy)",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

# Load latest model checkpoint for a given model name
def load_latest_model(model_name):
    try:
        model_base = "_".join(model_name.split("_")[:-2]) if "_lr" in model_name else model_name
        output_dir = "output"
        model_dir = sorted(
            [d for d in os.listdir(output_dir) if d.startswith(model_name)],
            reverse=True
        )[0]

        model_path = sorted(
            [f for f in os.listdir(f"{output_dir}/{model_dir}") if f.endswith(".pth")],
            key=lambda f: os.path.getmtime(f"{output_dir}/{model_dir}/{f}")
        )[-1]
        checkpoint_path = os.path.join(output_dir, model_dir, model_path)

        # Assign weights based on model
        if model_base == "resnet50":
            weights = "ResNet50_Weights.IMAGENET1K_V1"
        elif model_base == "efficientnet_v2_s":
            weights = "EfficientNet_V2_S_Weights.IMAGENET1K_V1"
        else:
            weights = "DEFAULT"

        model = get_model(model_base, weights=weights).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        model.eval()

        print(f"✅ Loaded model: {checkpoint_path}")
        return model

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise

# Preprocess uploaded retina image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

# Main prediction function
def predict(image_path, model_name):
    try:
        model = load_latest_model(model_name)
    except Exception as e:
        return {"error": str(e)}

    image = preprocess_image(image_path)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(output, 1).item()
        confidence = round(probabilities[predicted_class].item() * 100, 2)

    # Plot class probabilities
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=list(severity_labels.values()),
        y=probabilities.cpu().numpy(),
        ax=ax
    )
    ax.set_title(f"Class Probabilities for {severity_labels[predicted_class]}")
    ax.set_ylabel("Probability")
    ax.set_xlabel("DR Stage")
    plt.xticks(rotation=20)
    plt.tight_layout()
    chart_path = "outputs/probabilities_chart.png"
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(chart_path)
    plt.close()

    # Generate dummy confusion matrix
    dummy_cm = np.random.randint(10, 100, size=(5, 5))
    df_cm = pd.DataFrame(dummy_cm, index=severity_labels.values(), columns=severity_labels.values())

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Dummy)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    conf_matrix_path = "outputs/confusion_matrix.png"
    plt.savefig(conf_matrix_path)
    plt.close()

    return {
        "severity": severity_labels[predicted_class],
        "confidence": confidence,
        "probabilities": {
            severity_labels[i]: round(prob.item() * 100, 2) for i, prob in enumerate(probabilities)
        },
        "chart_path": chart_path,
        "conf_matrix_path": conf_matrix_path
    }

# For direct execution
if __name__ == "__main__":
    model_name = "mlp_mixer_lr0.003_bs16"  # Example
    test_image = "sample.jpg"              # Replace with your test file
    result = predict(test_image, model_name)

    if "error" in result:
        print(result["error"])
    else:
        print(f"Prediction: {result['severity']} ({result['confidence']}%)")
        print(json.dumps(result["probabilities"], indent=4))
