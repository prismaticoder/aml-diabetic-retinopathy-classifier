🚀 Diabetic Retinopathy Classifier

A deep learning-based tool to detect Diabetic Retinopathy (DR) using ResNet-50 and EfficientNet-B0 trained on fundus images. More models will be trained and added in future updates.

📌 Features

✔ Trained CNN Models – ResNet-50 & EfficientNet-B0✔ Image Upload via Web UI – Run inference on new retina images✔ Automatic Model Selection – Choose between different trained models✔ Performance Metrics – Accuracy, Confusion Matrix, and Probability Distributions✔ Scalable Training Pipeline – Continually train new models and update results

📂 Project Structure
```bash
diabetic_retinopathy_grading/
│── dataset/              # Dataset (⚠️ Not included in GitHub, download from Google Drive)
│── venv/                 # Virtual Environment (⚠️ Not included in GitHub, download from Google Drive)
│── output/               # Trained models & results (Check Google Drive for the latest models)
│── train.py              # Model Training Script
│── test.py               # Model Evaluation
│── inference.py          # Inference on New Images
│── model.py              # Model Architecture (Currently includes ResNet-50 & EfficientNet-B0; more models will be trained and added)
│── dataset.py            # Data Loading & Augmentation
│── app.py                # Streamlit Web UI for Deployment
│── README.md             # Documentation
│── requirements.txt      # Required Python Libraries
└── .gitignore            # Files to ignore when pushing to GitHub
```

🛠 Installation & Setup

1️⃣ Clone this repository

git clone https://github.com/prismaticoder/aml-diabetic-retinopathy-classifier.git
cd aml-diabetic-retinopathy-classifier

2️⃣ Download Dataset & Virtual Environment

The dataset, virtual environment, and pre-trained models are not included in GitHub due to size constraints. Download them from Google Drive:

🔗 Google Drive Link

https://drive.google.com/drive/folders/1-ZTJj6OCLdIkAK7dPKd4dLW4SpdRzjBF?usp=sharing

Extract the downloaded ZIP file.

Copy and paste the dataset/, venv/, and output/ folders into your cloned repository.

3️⃣ Activate Virtual Environment

For Windows:

venv\Scripts\activate

For Linux/Mac:

source venv/bin/activate

4️⃣ Install Dependencies

pip install -r requirements.txt

🚀 Running the Application

🖥 Train a Model

python train.py

🔎 Test Model Performance

python test.py

🌐 Launch the Streamlit Web App

streamlit run app.py

📌 Important Notes

The dataset and venv are too large to be pushed to GitHub. Always download them from the provided Google Drive link.

The output/ folder contains trained models and will be updated over time. Check Google Drive periodically for the latest trained models.

New models will be added to model.py as training progresses.

📢 Contribution

If you would like to contribute, feel free to fork the repository and submit a pull request! 🚀

🔗 References

Google Drive Download: Click Here

GitHub Repository: Click Here

🚀 Happy Coding! 🧑‍💻

