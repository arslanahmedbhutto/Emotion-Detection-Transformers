import os
import torch
import numpy as np
import tensorflow as tf

from flask import Flask, request, render_template
from transformers import (
    DistilBertTokenizer, TFDistilBertForSequenceClassification,
    XLNetTokenizer, TFXLNetForSequenceClassification,
    RobertaTokenizer, TFRobertaForSequenceClassification,
    AutoTokenizer, AutoModelForSequenceClassification
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)

# Load DistilBERT (TensorFlow)
DISTILBERT_DIR = os.path.join(BASE_DIR, 'models', 'distilbert')
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_DIR)
distilbert_model = TFDistilBertForSequenceClassification.from_pretrained(DISTILBERT_DIR)
distilbert_model.trainable = False

# Load XLNet (TensorFlow)
XLNET_DIR = os.path.join(BASE_DIR, 'models', 'xlnet')
xlnet_tokenizer = XLNetTokenizer.from_pretrained(XLNET_DIR)
xlnet_model = TFXLNetForSequenceClassification.from_pretrained(XLNET_DIR)
xlnet_model.trainable = False

# Load RoBERTa (TensorFlow)
ROBERTA_DIR = os.path.join(BASE_DIR, 'models', 'roberta')
roberta_tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_DIR)
roberta_model = TFRobertaForSequenceClassification.from_pretrained(ROBERTA_DIR)
roberta_model.trainable = False

# Load BigBird if available
BIGBIRD_DIR = os.path.join(BASE_DIR, 'models', 'bigbird')
bigbird_tokenizer = None
bigbird_model = None
bigbird_available = False
try:
    bigbird_tokenizer = AutoTokenizer.from_pretrained(BIGBIRD_DIR)
    bigbird_model = AutoModelForSequenceClassification.from_pretrained(BIGBIRD_DIR)
    bigbird_model.eval()
    bigbird_available = True
except Exception as exc:
    print(f"BigBird load failed: {exc}")

label_map = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'love', 4: 'sadness', 5: 'surprise'}


def predict_emotion(text, model_type):
    if model_type == 'distilbert':
        tokenizer = distilbert_tokenizer
        model = distilbert_model
        tensor_type = 'tf'
    elif model_type == 'xlnet':
        tokenizer = xlnet_tokenizer
        model = xlnet_model
        tensor_type = 'tf'
    elif model_type == 'roberta':
        tokenizer = roberta_tokenizer
        model = roberta_model
        tensor_type = 'tf'
    elif model_type == 'bigbird' and bigbird_available:
        tokenizer = bigbird_tokenizer
        model = bigbird_model
        tensor_type = 'pt'
    else:
        return "Invalid model type", None

    if tensor_type == 'tf':
        inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=128)
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
        predicted_class_index = int(np.argmax(probabilities))
        predicted_class = label_map[predicted_class_index]
        return predicted_class, probabilities
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
    predicted_class_index = int(np.argmax(predictions))
    predicted_class = label_map[predicted_class_index]
    return predicted_class, predictions


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        model_type = request.form['model_type']
        predicted_class, confidence = predict_emotion(text, model_type)
        return render_template('index.html', predicted_class=predicted_class, confidence=confidence, text=text, model_type=model_type)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run()
