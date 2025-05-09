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

severity_labels = {
    0: "No DR (Healthy)",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR (Most Severe)"
}

def load_latest_model(model_name):
    try:
        base_model_name = "_".join(model_name.split("_")[:-2])

        model_dir = sorted(
            [d for d in os.listdir("output") if d.startswith(model_name)],
            reverse=True
        )[0]
        model_path = sorted(
            [f for f in os.listdir(f"output/{model_dir}") if f.endswith(".pth")],
            key=lambda f: os.path.getmtime(f"output/{model_dir}/{f}")
        )[-1]
        full_path = os.path.join("output", model_dir, model_path)

        if base_model_name.lower() == "resnet50":
            weights = "ResNet50_Weights.IMAGENET1K_V1"
        elif base_model_name.lower() == "efficientnet_v2_s":
            weights = "EfficientNet_V2_S_Weights.IMAGENET1K_V1"
        else:
            weights = "DEFAULT"

        model = get_model(base_model_name.lower(), weights=weights).to(device)
        checkpoint = torch.load(full_path, map_location=device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        print(f"✅ Loaded model: {full_path}")
        return model

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image.to(device)

def predict(image_path, model_name):
    try:
        model = load_latest_model(model_name)
    except FileNotFoundError as e:
        return {"error": str(e)}

    image = preprocess_image(image_path)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(output, 1).item()
        confidence = probabilities[predicted_class].item() * 100

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(severity_labels.values()), y=probabilities.cpu().numpy(), ax=ax)
    ax.set_title(f"Class Probabilities for {severity_labels[predicted_class]}")
    ax.set_ylabel("Probability (%)")
    ax.set_xlabel("Class")
    ax.set_xticklabels(severity_labels.values(), rotation=20)
    plt.tight_layout()
    plt_path = "probabilities_chart.png"
    plt.savefig(plt_path)
    plt.close()

    # Simulated confusion matrix for UI
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

    return {
        "severity": severity_labels[predicted_class],
        "confidence": round(confidence, 2),
        "probabilities": {severity_labels[i]: round(prob.item() * 100, 2) for i, prob in enumerate(probabilities)},
        "chart_path": plt_path,
        "conf_matrix_path": conf_matrix_path
    }

if __name__ == "__main__":
    model_name = "efficientnet_v2_s_lr0.0001_bs32"
    test_image = "test_image.jpeg"
    result = predict(test_image, model_name)

    if "error" in result:
        print(result["error"])
    else:
        print(f"Predicted Severity: {result['severity']} (Confidence: {result['confidence']}%)")
        print("All probabilities:", json.dumps(result["probabilities"], indent=4))
