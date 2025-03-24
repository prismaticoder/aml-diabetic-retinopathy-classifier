# 🔍 This script contains a function that builds and returns a pre-trained model (ResNet-50 or EfficientNet-B0)
# 💡 We modify the last layer of the model so it can classify 5 types of Diabetic Retinopathy.

import torch
import torch.nn as nn
import torchvision.models as models  # 🧠 Import standard models like ResNet, EfficientNet from torchvision

# 🧠 This function takes a model name (like 'resnet50') and returns a modified version of that model
def get_model(model_name):
    # ✅ Make sure the model name is lowercase and doesn’t have extra spaces
    model_name = model_name.lower().strip()

    # 🛠️ Fix common naming typos (e.g., someone writes "efficientnet" instead of "efficientnet_b0")
    model_mapping = {
        "efficientnet": "efficientnet_b0",
        "efficientnet_b0": "efficientnet_b0",
        "resnet50": "resnet50"
    }

    # ❌ If the user enters an invalid model name, we stop and raise an error
    if model_name not in model_mapping:
        raise ValueError(f"❌ '{model_name}' is not supported! Please use one of: resnet50, efficientnet_b0")

    # ✅ Use the corrected name
    corrected_model_name = model_mapping[model_name]

    # 📦 Get the actual model function (e.g., models.resnet50)
    model_class = getattr(models, corrected_model_name, None)

    # ❌ Check if model class exists and is callable
    if model_class is None or not callable(model_class):
        raise ValueError(f"❌ '{corrected_model_name}' cannot be loaded from torchvision!")

    # 🧠 Try to load the model with pre-trained weights (ImageNet-trained)
    try:
        weights = None  # Default to None for safety

        if corrected_model_name == "efficientnet_b0":
            # 📥 EfficientNet weights
            from torchvision.models.efficientnet import EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.DEFAULT

        elif corrected_model_name == "resnet50":
            # 📥 ResNet50 weights
            from torchvision.models.resnet import ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT

        # 📦 Instantiate the model with pre-trained weights
        model = model_class(weights=weights)

    except TypeError:
        # ⏳ Fallback for older versions of torchvision that use pretrained=True
        model = model_class(pretrained=True)

    # 🧩 Now we need to change the final layer so the model outputs 5 classes instead of 1000
    if hasattr(model, "fc"):  # ✅ For models like ResNet or DenseNet
        num_ftrs = model.fc.in_features  # 🔢 Get the number of features going into the final layer
        model.fc = nn.Linear(num_ftrs, 5)  # 🧠 Replace with new layer for 5-class classification

    elif hasattr(model, "classifier"):  # ✅ For models like EfficientNet or VGG
        num_ftrs = model.classifier[-1].in_features  # 🔢 Get input features from last layer
        model.classifier[-1] = nn.Linear(num_ftrs, 5)  # 🔁 Replace the last layer with our 5-class layer

    else:
        # ❌ If the model structure is not recognized
        raise ValueError(f"❌ Can’t modify output layer for {corrected_model_name}")

    # ✅ Return the updated model ready to train or use for inference
    return model
