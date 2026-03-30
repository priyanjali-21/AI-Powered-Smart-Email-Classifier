# 🚀 AI Email Classifier

An AI-powered system that automatically classifies emails and detects their urgency using DistilBERT and simple rule-based logic.

---

## 📌 Overview

This project helps manage emails efficiently by analyzing their content and predicting:

- 📂 **Category** → Complaint, Request, Feedback, Spam  
- ⚡ **Urgency Level** → High, Medium, Low  

It uses two separate models:
- One for **category classification**
- One for **urgency detection**

This improves accuracy and makes the system scalable.

---

## 🧠 Features

- 🔍 Automatic email classification  
- 🤖 Uses DistilBERT for better text understanding  
- ⚙️ Rule-based urgency detection  
- 🌐 FastAPI backend  
- 🎨 Streamlit frontend dashboard  
- 📊 Simple analytics and visualization  

---

## 🏗️ Project Structure
AI-Email-Classifier/
│
├── backend/
│ └── app.py
│
├── frontend/
│ └── app.py
│
├── models/
│ ├── category_model/
│ └── urgency_model/
│
├── requirements.txt
├── README.md
└── .gitignore

---

## 🔄 Workflow

1. User enters email text in the frontend  
2. Request is sent to FastAPI backend  
3. Text is processed using tokenizer  
4. Models predict:
   - Category (DistilBERT)  
   - Urgency (DistilBERT + rules)  
5. Results are returned and shown on dashboard  

---

## 🛠️ Tech Stack

- Python  
- FastAPI  
- Streamlit  
- Hugging Face Transformers (DistilBERT)  
- Scikit-learn  
- Pandas  

---

