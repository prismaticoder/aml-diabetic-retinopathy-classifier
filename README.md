🚀 Diabetic Retinopathy Classifier

A deep learning-based tool to detect Diabetic Retinopathy (DR) using ResNet-50 and EfficientNet-B0 trained on fundus images. More models will be trained and added in future updates.

👥 Team Setup Instructions

1️⃣ Download Dataset

- Access the refined_dataset.zip from [Google Drive](https://drive.google.com/drive/folders/1-ZTJj6OCLdIkAK7dPKd4dLW4SpdRzjBF?usp=sharing)
- Extract the ZIP file
- Place the extracted `dataset` folder in your project's root directory

2️⃣ Set Up Virtual Environment

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For Linux/Mac
python -m venv venv
source venv/bin/activate
```

3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

4️⃣ Training Process

```bash
python train.py
```

- Select your name (1-4):
  1. Zohaib
  2. Larry
  3. Meena
  4. Tom
  5. Reviewer
- Enter the model name (resnet50, efficientnet_b0, etc.)
- Configure batch size (optional, default=32)
- Set learning rate (optional, default=0.0001)

5️⃣ Logging & Version Control

- Training logs are automatically saved in the `logs` directory
- Commit and push your logs to share results with the team
- Log files are named: `{username}_{model}_{timestamp}.json`

📂 Project Structure

```bash
diabetic_retinopathy_grading/
│── dataset/              # Dataset folder (extract from refined_dataset.zip)
│── logs/                 # Training logs directory
│── output/               # Trained models & results
│── train.py              # Model Training Script
│── test.py               # Model Evaluation
│── inference.py          # Inference on New Images
│── model.py              # Model Architecture
│── dataset.py            # Data Loading & Augmentation
│── app.py                # Streamlit Web UI for Deployment
│── README.md             # Documentation
│── requirements.txt      # Required Python Libraries
└── .gitignore            # Files to ignore when pushing to GitHub
```

📌 Features

✔ Trained CNN Models – ResNet-50 & EfficientNet-B0
✔ Image Upload via Web UI – Run inference on new retina images
✔ Automatic Model Selection – Choose between different trained models
✔ Performance Metrics – Accuracy, Confusion Matrix, and Probability Distributions
✔ Scalable Training Pipeline – Continually train new models and update results

📢 Contribution

If you would like to contribute, feel free to fork the repository and submit a pull request! 🚀

🔗 References

Google Drive Download: [Click Here](https://drive.google.com/drive/folders/1-ZTJj6OCLdIkAK7dPKd4dLW4SpdRzjBF?usp=sharing)

GitHub Repository: [Click Here](https://github.com/prismaticoder/aml-diabetic-retinopathy-classifier.git)

🚀 Happy Coding! 🧑‍💻
