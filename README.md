# 🌟 AI-Powered Image Classification System  

**👨‍💻 Developer:** Arun (GitHub: [9696arun](https://github.com/9696arun))  
**🏢 Organization:** Flikt Technology Web Solution  
**📅 Project Year:** 2025  
**🎯 Assignment:** AI Developer Technical Project — Deep Learning Image Classification  

---

## 🚀 Overview  

This project is an **AI-powered image classification system** built using **Deep Learning (CNN)** and deployed through a **Streamlit web interface**.  
The model can automatically classify images into **five categories** — **Birds, Cats, Dogs, Fruits, and Tiger/Lion** — with high accuracy and real-time performance.

It is designed as part of the **AI Developer Training Program** at **Flikt Technology Web Solution**, focusing on:
- Model building and optimization  
- Data preprocessing and visualization  
- Deployment readiness and user interaction  

---

## 🧠 Objectives  

- Build and train a **Convolutional Neural Network (CNN)** for multi-class image classification.  
- Evaluate model accuracy, precision, recall, and F1-score.  
- Visualize training and testing performance.  
- Create a **user-friendly Streamlit web app** for real-time predictions.  

---

## 📂 Dataset Information  

A **custom dataset** was used for training and testing with 5 image classes:  

| Class | Description |
|:------|:-------------|
| 🐦 **Birds** | Different species of birds |
| 🐱 **Cats** | Domestic cat images |
| 🐶 **Dogs** | Multiple dog breeds |
| 🍎 **Fruits** | Apples, bananas, oranges, etc. |
| 🦁 **Tiger/Lion** | Wild big cats |

**Dataset Split:**
- Training Set → 70%  
- Validation Set → 15%  
- Testing Set → 15%

Total images: **1000+ labeled samples**

---

## 🏗️ Model Architecture  

Developed using **TensorFlow/Keras**, this CNN model includes:  

- 3 × **Convolutional Layers**  
- **Batch Normalization** & **Dropout Layers**  
- **ReLU Activation Functions**  
- **MaxPooling2D Layers**  
- **Adam Optimizer**  
- **Categorical Crossentropy Loss Function**  
- **Early Stopping Callback**

**Input Shape:** `(150, 150, 3)`  
**Output Classes:** `5`

---

## 📊 Model Evaluation  

| Metric | Score |
|:--------|:------|
| ✅ Accuracy | 92% |
| 🎯 Precision | 90% |
| 📈 Recall | 89% |
| 🧮 F1-Score | 89.5% |

**Performance Visualizations:**  
- Training vs. Validation Accuracy Curve  
- Training vs. Validation Loss Curve  
- Confusion Matrix Visualization  

---

## 💻 Streamlit Web Application  

An interactive **web-based interface** built using **Streamlit** allows users to upload an image and instantly view predictions.  

### 🖼️ Sample Prediction Output  

Below are real screenshots of the web application’s working and prediction results 👇  

#### 🔹 Image Upload & Prediction Interface  
![App Interface](output/image.png)

#### 🔹 Predicted Output Example 1  
![Output Screenshot 1](output/Screenshot%202025-11-07%20005425.png)

#### 🔹 Predicted Output Example 2  
![Output Screenshot 2](output/Screenshot%202025-11-07%20005443.png)

#### 🔹 Predicted Output Example 3  
![Output Screenshot 3](output/Screenshot%202025-11-07%20005853.png)

---

## 🧩 Features  

- 🖼️ Upload images (`.jpg`, `.jpeg`, `.png`)  
- 📊 Instant classification results  
- 🎨 Clean, responsive UI design  
- ⚡ Real-time predictions  
- 💾 Uses trained CNN model (`best_model.h5`)  

---

## ▶️ Run the App  

```bash
streamlit run app.py
