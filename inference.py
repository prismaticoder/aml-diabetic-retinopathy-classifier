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
        model_dir = sorted(
            [d for d in os.listdir("output") if d.startswith(model_name)],
            reverse=True
        )[0]

        checkpoint_path = sorted(
            [f for f in os.listdir(f"output/{model_dir}") if f.endswith(".pth")],
            key=lambda f: os.path.getmtime(os.path.join("output", model_dir, f))
        )[-1]

        full_path = os.path.join("output", model_dir, checkpoint_path)

        weights = "DEFAULT"
        model = get_model(model_name.lower(), weights=weights, n_classes=5).to(device)
        checkpoint = torch.load(full_path, map_location=device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
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
    ax.set_title(f"Prediction: {severity_labels[predicted_class]} ({confidence:.2f}%)")
    ax.set_ylabel("Probability")
    ax.set_xticklabels(severity_labels.values(), rotation=20)
    plt.tight_layout()
    chart_path = "probabilities_chart.png"
    plt.savefig(chart_path)
    plt.close()

    # Dummy Confusion Matrix
    dummy_cm = np.random.randint(5, 20, size=(5, 5))
    df_cm = pd.DataFrame(dummy_cm, index=severity_labels.values(), columns=severity_labels.values())
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Dummy Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    cm_path = "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    return {
        "severity": severity_labels[predicted_class],
        "confidence": round(confidence, 2),
        "probabilities": {severity_labels[i]: round(prob.item() * 100, 2) for i, prob in enumerate(probabilities)},
        "chart_path": chart_path,
        "conf_matrix_path": cm_path
    }

# Optional CLI test
if __name__ == "__main__":
    model_name = "mlp_mixer_lr0.0001_bs32"
    test_image = "test_image.jpg"
    result = predict(test_image, model_name)

    if "error" in result:
        print(result["error"])
    else:
        print(f"Predicted Severity: {result['severity']} (Confidence: {result['confidence']}%)")
        print(json.dumps(result["probabilities"], indent=4))
