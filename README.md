# 🍃 Deep Learning-Based Plant Disease Identification Using CNN & Transfer Learning

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Applications-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Status](https://img.shields.io/badge/Project-Completed-success?style=for-the-badge)

An end-to-end Deep Learning pipeline built with TensorFlow/Keras to identify tomato leaf diseases from the PlantVillage dataset. This repository covers data preprocessing, custom Baseline CNN construction, MobileNetV2 Transfer Learning (Feature Extraction + Fine-tuning), comprehensive metric evaluation, and qualitative error analysis.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Summary](#-dataset-summary)
- [System Architecture & Pipeline](#-system-architecture--pipeline)
- [Performance & Results](#-performance--results)
- [Deep Error Analysis](#-deep-error-analysis)
- [Repository Structure](#-repository-structure)
- [Installation & Usage](#-installation--usage)
- [Future Work](#-future-work)
- [Authors & Acknowledgments](#-authors--acknowledgments)

---

## 📖 Project Overview

Plant diseases represent a major threat to global agricultural yield and food security. Early detection using automated Computer Vision models allows farmers to intervene before severe crop loss occurs.

This project implements and compares two deep learning approaches to classify **5 distinct tomato leaf conditions**:
1. **Custom Baseline CNN**: Built from scratch with 3 convolutional blocks.
2. **MobileNetV2 Transfer Learning**: Pre-trained on ImageNet, fine-tuned in two strategic phases.

---

## 📊 Dataset Summary

The dataset is derived from the **PlantVillage Dataset**, specifically focused on tomato crop leaves.

* **Total Training Images:** 5,780 images (80%)
* **Total Validation Images:** 1,443 images (20%)
* **Target Size:** $224 \times 224$ pixels
* **Number of Classes:** 5 classes

| Class Index | Folder Name | Short Name |
| :---: | :--- | :--- |
| `0` | `Tomato_Early_blight` | Early Blight |
| `1` | `Tomato_Late_blight` | Late Blight |
| `2` | `Tomato_Leaf_Mold` | Leaf Mold[cite: 1] |
| `3` | `Tomato_Septoria_leaf_spot` | Septoria L.S.[cite: 1] |
| `4` | `Tomato_healthy` | Healthy[cite: 1] |

### Preprocessing & Data Augmentation
To mitigate overfitting and increase feature robustness, the following augmentations were applied[cite: 1]:
* **Rescaling:** $1/255$[cite: 1]
* **Rotation Range:** $20^\circ$[cite: 1]
* **Horizontal Flip:** `True`[cite: 1]
* **Zoom Range:** $0.2$[cite: 1]

---

## 🏗️ System Architecture & Pipeline

Raw Images ──► Augmentation & Preprocessing ──► Model Training ──► Evaluation & Error Analysis
│
┌────────────────────┴────────────────────┐
▼                                         ▼
Baseline CNN                             MobileNetV2
(3 Conv Blocks)                     (Feature Ext. + Fine-Tune)

### 1. Custom Baseline CNN
* **Architecture:** 3x `Conv2D` + `ReLU` + `MaxPooling2D` blocks $\rightarrow$ `Flatten` $\rightarrow$ `Dense(128)` + `Dropout(0.5)` $\rightarrow$ `Dense(5, Softmax)`[cite: 1].
* **Optimizer:** Adam ($\text{lr} = 10^{-3}$)[cite: 1].
* **Loss Function:** Categorical Crossentropy[cite: 1].

### 2. MobileNetV2 Transfer Learning
* **Phase 1 (Feature Extraction):** Base model completely frozen (`trainable = False`), custom classification head trained at $\text{lr} = 10^{-4}$[cite: 1].
* **Phase 2 (Fine-tuning):** Unfrozen top 30 layers of MobileNetV2 base model, trained at a lower learning rate ($\text{lr} = 10^{-5}$)[cite: 1].
* **Callbacks:** `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint`[cite: 1].

---

## 📈 Performance & Results

### Model Comparison Table

| Metric | Custom Baseline CNN | MobileNetV2 (Fine-tuned) | Delta ($\uparrow$) |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy** | **81.08%**[cite: 1] | **92.24%**[cite: 1] | **+11.16%**[cite: 1] |
| **Weighted F1-Score** | 80.78%[cite: 1] | **92.01%**[cite: 1] | +11.24%[cite: 1] |
| **Macro F1-Score** | 79.50%[cite: 1] | **90.75%**[cite: 1] | +11.25%[cite: 1] |
| **Weighted Precision** | 84.21%[cite: 1] | **92.61%**[cite: 1] | +08.39%[cite: 1] |
| **Weighted Recall** | 81.08%[cite: 1] | **92.24%**[cite: 1] | +11.16%[cite: 1] |

### MobileNetV2 Class-wise Metrics

precision    recall  f1-score   support
Early Blight     0.9384    0.6850    0.7919       200
Late Blight     0.9538    0.9213    0.9372       381
Leaf Mold     0.8139    0.9895    0.8931       190
Septoria L.S.     0.9158    0.9520    0.9335       354
Healthy     0.9636    1.0000    0.9815       318
accuracy                         0.9224      1443

---

## 🔍 Deep Error Analysis

Out of **1,443 validation samples**, MobileNetV2 correctly predicted **1,331 samples (92.24%)** and misclassified **112 samples (7.76%)**[cite: 1].

### Key Findings:
1. **Early Blight Bottleneck:** Early Blight exhibited the highest error rate (**31.50%** misclassified)[cite: 1]. Most misclassifications occurred with **Septoria Leaf Spot** (26 instances) and **Leaf Mold** (16 instances) due to overlapping lesion morphologies during early infection phases[cite: 1].
2. **Healthy Leaf Perfection:** The model achieved **100% recall** on the `Healthy` class (0 misclassifications)[cite: 1].
3. **Confidence Distribution:** Average confidence for correct predictions was **96.01%**, whereas incorrect predictions had a lower average confidence of **70.90%**[cite: 1]. However, 41 samples were high-confidence errors ($>80\%$), mainly caused by strong shadow masking and background noise[cite: 1].

---

## 📁 Repository Structure

```text
├── models/                         # Saved Keras model files (.keras)
│   ├── baseline_cnn.keras
│   └── mobilenetv2_final.keras
├── results/                        # Generated evaluation charts & figures
│   └── comparison_baseline_vs_mobilenet.png
├── plant_disease_classification.ipynb # Main execution Jupyter Notebook
├── README.md                       # Project Documentation
└── requirements.txt                # Python Dependencies
```

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/Dalangans/Deep-Learning-Plant-Disease-Identification.git
cd Deep-Learning-Plant-Disease-Identification
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/MacOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Dependencies (requirements.txt)
```Plaintext
tensorflow>=2.11.0
numpy
pandas
matplotlib
seaborn
scikit-learn
pillow
```

### 4. Run the Notebook
Launch Jupyter Lab or VS Code to run plant_disease_classification.ipynb:

```bash
jupyter lab
```

---

## 🚀 Future Work

* **Focal Loss Integration:** Replace standard Categorical Cross-Entropy with Focal Loss to address hard negative samples (e.g., Early Blight).
* **Contrast & Shadow Preprocessing:** Integrate CLAHE (Contrast Limited Adaptive Histogram Equalization) and shadow augmentation to minimize lighting interference.
* **Model Explainability (Grad-CAM):** Implement Grad-CAM visual attention maps to highlight regions of interest leading to model decisions.
* **Isolated Hold-out Test Set:** Establish a strict, isolated test dataset prior to image generators to ensure zero data leakage.

---

## 👥 Authors & Acknowledgments

### Development Team (Group Project)
* **Nabiel Harits Utomo** (NPM: 2306267044)
* **Muhammad Pavel** (NPM: 2306242363)
* **R. Aisha Syauqi Ramadhani** (NPM: 2306250554)
* **Zhafira Zahra Alfarisy** (NPM: 2306250636)

### Course Details
* **Course:** Artificial Intelligence (Kecerdasan Buatan) — Final Project
* **Department:** Computer Engineering, Universitas Indonesia
* **Advisor/Lecturer:** Muhammad Firdaus Syawaludin Lubis, S.T., M.T., Ph.D.
