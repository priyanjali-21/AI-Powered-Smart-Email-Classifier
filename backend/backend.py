from fastapi import FastAPI
from pydantic import BaseModel

import torch
import pickle
from transformers import DistilBertForSequenceClassification, AutoTokenizer

# =========================
# INIT
# =========================
app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODELS
# =========================

tokenizer = AutoTokenizer.from_pretrained("/home/capt-praveen/Desktop/Ai_Email_Classifier/category_model")

model_cat = DistilBertForSequenceClassification.from_pretrained("/home/capt-praveen/Desktop/Ai_Email_Classifier/category_model").to(device)
model_urg = DistilBertForSequenceClassification.from_pretrained("/home/capt-praveen/Desktop/Ai_Email_Classifier/urgency_model").to(device)

model_cat.eval()
model_urg.eval()

with open("/home/capt-praveen/Desktop/Ai_Email_Classifier/category_model/category_encoder.pkl", "rb") as f:
    category_encoder = pickle.load(f)

with open("/home/capt-praveen/Desktop/Ai_Email_Classifier/urgency_model/urgency_encoder.pkl", "rb") as f:
    urgency_encoder = pickle.load(f)

# =========================
# RULE BASED
# =========================

urgent_keywords = ["urgent", "asap", "immediately", "server down", "critical"]

def rule_based_urgency(text):
    text = text.lower()
    for word in urgent_keywords:
        if word in text:
            return "high"
    return None

def final_urgency(text, model_pred):
    rule = rule_based_urgency(text)
    if rule:
        return rule
    return urgency_encoder.inverse_transform([model_pred])[0]


def rule_based_complaint(text):
    text = text.lower()

    strong_phrases = ["refund", "not working", "worst", "unacceptable", "frustrated"]
    medium_words = ["issue", "problem", "error", "bug", "complaint"]

    score = 0

    for w in strong_phrases:
        if w in text:
            score += 3

    for w in medium_words:
        if w in text:
            score += 2

    if score >= 3:
        return "complaint"

    return None

# =========================
# REQUEST SCHEMA
# =========================

class EmailRequest(BaseModel):
    email: str

# =========================
# API
# =========================

@app.post("/classify")
def classify_email(req: EmailRequest):

    text = req.email.lower().strip()

    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # CATEGORY
    with torch.no_grad():
        out_cat = model_cat(input_ids=input_ids, attention_mask=attention_mask)
        probs_cat = torch.softmax(out_cat.logits, dim=1)
        conf_cat, pred_cat = torch.max(probs_cat, dim=1)

    ml_category = category_encoder.inverse_transform([pred_cat.item()])[0]

    # HYBRID CATEGORY
    rule_cat = rule_based_complaint(text)

    if rule_cat:
        category = "complaint"
        category_conf = 0.95
    else:
        category = ml_category
        category_conf = conf_cat.item()

    # URGENCY
    with torch.no_grad():
        out_urg = model_urg(input_ids=input_ids, attention_mask=attention_mask)
        probs_urg = torch.softmax(out_urg.logits, dim=1)
        conf_urg, pred_urg = torch.max(probs_urg, dim=1)

    urgency = final_urgency(text, pred_urg.item())

    # 🔥 SINGLE CONFIDENCE (frontend requirement)
    final_conf = max(category_conf, conf_urg.item())

    return {
        "category": category,
        "urgency": urgency,
        "confidence": round(final_conf, 3)
    }