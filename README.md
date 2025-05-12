# Diabetic Retinopathy Grading – Final Submission Branch

This repository contains the final submission for the EEEM068 Applied Machine Learning project on **Diabetic Retinopathy Grading**. It includes custom transformer-based architectures, detailed experiment tracking, ablation studies, and a visual demo dashboard powered by Streamlit.

> **Note:** This is the **submission branch**. All individual work and development occurred in separate branches but has been merged and organized here for evaluation purposes.

---

## 📁 Folder Structure

```bash
├── models/                  # Contains implementations for all model architectures
│   ├── maxvit.py
│   ├── swin_transformer.py
│   ├── rsgnet.py
│   ├── mlp_mixer.py
│   └── efficientnetv2.py   # Baseline model
│
├── experiments/            # Contains all experiments, organized by team member
│   ├── baseline/           # Contains baseline experiments
│   ├── 6891xx_zohaib/
│   │   ├── experiment_1.sh
│   │   └── ...
│   ├── 6904xx_meena/
│   │   ├── experiment_1.sh
│   │   └── ...
│   └── ...
│
├── dataset/                # Dataset folder (gitignored by default)
│   ├── train/              # Images folder
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── labels/                 # Where split_data.py initially writes train/val/test labels
│   └── ...
│
├── outputs/                # Outputs from each experiment (e.g., models, metrics)
│
├── logs/                   # Log files from training runs
│
├── app.py                  # Streamlit dashboard for prediction and explanations
├── split_data.py           # Script to perform stratified split of labels
├── requirements.txt
└── README.md
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/prismaticoder/aml-diabetic-retinopathy-classifier.git
cd aml-diabetic-retinopathy-classifier
git checkout submission
```

### 2. Install Dependencies

We recommend using a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Dataset Setup

Download the dataset from Kaggle:
[Kaggle DR Dataset](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data)

Unzipping the downloaded file should automatically create a `dataset/` folder. If it doesn't, create one manually and move the files accordingly.

Make sure the images are stored as:

```bash
dataset/train/0cd1683b5.jpg
```

---

## 📊 Available Models

We implemented five models:

* **MaxViT**: Combines CNN-like locality with transformer-style attention. We experimented with adding residual connections and removing block attention for ablation.
* **Swin Transformer**: Hierarchical transformer using shifted windows. We explored replacing patch merging with CNN + pooling layers.
* **RSGNet**: Residual Self-Gated Network that balances complexity and performance.
* **MLP Mixer**: Uses MLP layers instead of convolutions or attention. Provides an interesting contrast in architecture.
* **EfficientNetV2**: Used as our baseline model for performance comparison.

All model definitions are found in the `models/` directory.

---

## 🔢 Running Experiments

Each team member’s experiments are located in their respective folders inside the `experiments/` directory, named using the format `URN_name`.

To run a specific experiment:

```bash
cd experiments/6904186_meena
bash experiment_1.sh
```

After running, results are saved to:

* `outputs/` — for model checkpoints and evaluation metrics
* `logs/` — for training logs and console outputs

---

## 📊 Data Splitting

Run the following to split the dataset labels into training, validation, and test sets with stratification:

```bash
python split_data.py
```

This will generate `train.csv`, `val.csv`, and `test.csv` in the `labels/` directory. Move these files into the `dataset/` folder to enable training.

---

## 🚀 GPU Recommendation

We strongly recommend running this project with CUDA and a high-memory GPU due to the complexity and size of the transformer-based models.

---

## 💻 Streamlit Dashboard

You can launch the Streamlit-based frontend to interact with the trained model and view predictions:

```bash
streamlit run app.py
```

The dashboard allows you to:

* Upload a retinal image
* Predict the DR grade
* Receive actionable recommendations based on the result

---

## 🔝 Key Ablation Study Highlights

Throughout this project, we conducted multiple ablation studies to better understand model behavior and performance drivers:

* **MaxViT**: Removing block attention unexpectedly improved results; residual layers helped mitigate vanishing gradients.
* **Swin Transformer**: CNN-based patch merging improved accuracy over default transformer merging.
* **Warmup + Cosine Annealing**: Helped the models train longer without premature convergence.
* **Gradient Clipping and Weight Decay**: Improved generalization in deeper models.

---

## 👥 Contributors

* **ATTOH, LAWRENCE**
* **PREM, MEENAKSHY**
* **SALAM, JESUTOMIWA**
* **SHAIKH, ZOHAIB**

Thank you!