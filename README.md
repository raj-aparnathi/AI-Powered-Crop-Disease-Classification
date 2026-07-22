# 🌱 CropX - AI Powered Crop Disease Detection System

<p align="center">
  <img src="assets/logo.png" width="180" alt="CropX Logo">
</p>

<p align="center">
  <b>Deep Learning based Crop Disease Classification using MobileNetV3</b><br>
  Detect crop diseases from leaf images with high accuracy using TensorFlow & Streamlit.
</p>

---

## 📌 Overview

CropX is an AI-powered crop disease detection system that classifies plant diseases from leaf images using a **MobileNetV3** deep learning model.

The project supports **7 major crops** with **56 disease classes** and provides a lightweight solution that can be deployed as a **Streamlit Web Application**, **FastAPI Backend**, and **Flutter Mobile App**.

---

## 🚀 Features

- 🌿 7 Crop Categories
- 🦠 56 Disease Classes
- 🧠 MobileNetV3 Transfer Learning
- 📷 Image Upload & Prediction
- 🎯 Top-3 Prediction Results
- 📊 Confidence Score
- ⚡ Lightweight & Fast
- 📱 Flutter Compatible
- 🌐 Streamlit Web Interface
- 🔧 Modular Code Structure

---

# 🌾 Supported Crops

- Cotton
- Groundnut
- Maize
- Onion
- Potato
- Tomato
- Wheat

---

# 📂 Dataset

### Total Images

**62,000+ Images**

Dataset contains healthy and diseased leaf images collected from multiple public agricultural datasets.

### Dataset Link

https://drive.google.com/drive/folders/1F5tLCDoLxGh6URbTsU5M_kzKtQ73LrlW

---

# 🧠 Model Information

| Property | Value |
|----------|-------|
| Model | MobileNetV3 |
| Framework | TensorFlow / Keras |
| Transfer Learning | ✅ |
| Input Size | 224 × 224 |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Classes | 56 |
| Crops | 7 |

---

# 📊 Model Performance

| Metric | Score |
|---------|--------|
| Test Accuracy | **86.31%** |
| Test Loss | **0.4465** |
| Test Top-3 Accuracy | **96.86%** |

---

# 📁 Project Structure

```text
CropXSample/
│
├── app.py
├── utils.py
├── class_names.py
├── requirements.txt
│
├── model/
│   └── CropX_MobileNetV3_Best.keras
│
├── assets/
│
├── sample_images/
│
└── notebook/
    └── CropX_Model.ipynb
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/CropX.git

cd CropX
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

---

# 🖼️ Usage

1. Open the Streamlit application.
2. Upload a crop leaf image.
3. Click **Predict**.
4. The model will display:
   - Crop Name
   - Disease Name
   - Confidence Score
   - Top-3 Predictions

---

# 🧪 Model Pipeline

```
Leaf Image
      │
      ▼
Image Upload
      │
      ▼
Resize (224×224)
      │
      ▼
Normalization
      │
      ▼
MobileNetV3
      │
      ▼
Softmax Layer
      │
      ▼
Predicted Disease
```

---

# 📦 Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV3
- Streamlit
- NumPy
- OpenCV
- Pillow
- Matplotlib
- Scikit-learn

---

# 📈 Training Strategy

- Transfer Learning
- MobileNetV3 Backbone
- Fine-Tuning
- Early Stopping
- Model Checkpoint
- ReduceLROnPlateau
- Class Weight Balancing

---

# 📷 Sample Prediction

```
Crop:
Tomato

Disease:
Early Blight

Confidence:
98.72%

Top-3 Predictions

1. Early Blight (98.72%)
2. Leaf Mold (0.81%)
3. Septoria Leaf Spot (0.47%)
```

---

# 🔮 Future Improvements

- Disease Description
- Symptoms
- Chemical Treatment
- Organic Treatment
- Fertilizer Recommendation
- Weather Integration
- Farmer Dashboard
- Flutter Mobile Deployment
- FastAPI REST API
- TensorFlow Lite Deployment

---

# 🤝 Contributing

Contributions are welcome!

Feel free to fork this repository and submit a Pull Request.

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

---

# 📜 License

This project is intended for educational and research purposes.

---

## 🙏 Acknowledgements

- TensorFlow
- Keras
- Google
- PlantVillage Dataset
- Agricultural Open Datasets

---

<p align="center">
<b>🌱 CropX — AI for Smarter Agriculture 🌾</b>
</p>
