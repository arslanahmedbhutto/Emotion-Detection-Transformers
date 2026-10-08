"""
Interactive testing script for all transformer emotion classification models
Tests: DistilBERT, XLNet, RoBERTa, and BigBird
"""

import os
import torch
import numpy as np
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    XLNetTokenizer, XLNetForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    AutoTokenizer, AutoModelForSequenceClassification
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Emotion labels (adjust if your models use different labels)
LABEL_MAP = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'love', 4: 'sadness', 5: 'surprise'}

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}\n")

class ModelWrapper:
    """Wrapper for consistent model inference"""
    def __init__(self, name, tokenizer_path, model_path):
        self.name = name
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.loaded = False
        
    def load(self):
        try:
            print(f"Loading {self.name}...", end=" ")
            if self.name == "BigBird":
                # BigBird uses AutoTokenizer/AutoModel
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            # DistilBERT, XLNet and RoBERTa were trained in TensorFlow (tf_model.h5)
            elif self.name == "DistilBERT":
                self.tokenizer = DistilBertTokenizer.from_pretrained(self.tokenizer_path)
                self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path, from_tf=True)
            elif self.name == "XLNet":
                self.tokenizer = XLNetTokenizer.from_pretrained(self.tokenizer_path)
                self.model = XLNetForSequenceClassification.from_pretrained(self.model_path, from_tf=True)
            elif self.name == "RoBERTa":
                self.tokenizer = RobertaTokenizer.from_pretrained(self.tokenizer_path)
                self.model = RobertaForSequenceClassification.from_pretrained(self.model_path, from_tf=True)
            
            self.model.to(DEVICE)
            self.model.eval()
            self.loaded = True
            print("✓ Loaded")
            return True
        except Exception as e:
            print(f"✗ Failed: {e}")
            self.loaded = False
            return False
    
    def predict(self, text, max_length=128):
        if not self.loaded:
            return None, None
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            predicted_label = LABEL_MAP.get(predicted_class_idx, "unknown")
            confidence = probabilities[0][predicted_class_idx].item()
            
            return predicted_label, {
                'confidence': confidence,
                'probabilities': {LABEL_MAP.get(i, 'unknown'): float(p) 
                                for i, p in enumerate(probabilities[0].cpu().numpy())}
            }
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, None


def load_all_models():
    """Load all available models"""
    models = {}
    
    model_configs = [
        ("DistilBERT", 
         os.path.join(BASE_DIR, 'models', 'distilbert'),
         os.path.join(BASE_DIR, 'models', 'distilbert')),
        ("XLNet", 
         os.path.join(BASE_DIR, 'models', 'xlnet'),
         os.path.join(BASE_DIR, 'models', 'xlnet')),
        ("RoBERTa", 
         os.path.join(BASE_DIR, 'models', 'roberta'),
         os.path.join(BASE_DIR, 'models', 'roberta')),
        ("BigBird", 
         os.path.join(BASE_DIR, 'models', 'bigbird'),
         os.path.join(BASE_DIR, 'models', 'bigbird')),
    ]
    
    for name, tokenizer_path, model_path in model_configs:
        if os.path.exists(model_path):
            wrapper = ModelWrapper(name, tokenizer_path, model_path)
            if wrapper.load():
                models[name] = wrapper
        else:
            print(f"Skipping {name}: Model path not found at {model_path}")
    
    return models


def print_results(text, predictions):
    """Pretty print prediction results"""
    print(f"\nText: \"{text}\"")
    print("-" * 80)
    
    for model_name, (emotion, stats) in predictions.items():
        if emotion is None:
            print(f"  {model_name}: Failed to predict")
        else:
            print(f"  {model_name}:")
            print(f"    Emotion: {emotion.upper()} (Confidence: {stats['confidence']:.4f})")
            prob_str = " | ".join([f"{label}: {prob:.4f}" for label, prob in stats['probabilities'].items()])
            print(f"    Probabilities: {prob_str}")
    print()


def main():
    print("=" * 80)
    print("EMOTION CLASSIFICATION - LIVE MODEL TESTING")
    print("=" * 80)
    print()
    
    # Load all models
    models = load_all_models()
    
    if not models:
        print("ERROR: No models were successfully loaded!")
        return
    
    print(f"\nSuccessfully loaded {len(models)} model(s): {', '.join(models.keys())}")
    print("=" * 80)
    print("Enter text to classify (type 'quit' to exit, 'clear' to clear screen)\n")
    
    while True:
        try:
            text = input("Enter emotion text: ").strip()
            
            if text.lower() == 'quit':
                print("\nGoodbye!")
                break
            
            if text.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print("EMOTION CLASSIFICATION - LIVE MODEL TESTING\n")
                continue
            
            if not text:
                print("Please enter some text.\n")
                continue
            
            # Test all models
            predictions = {}
            for model_name, model in models.items():
                emotion, stats = model.predict(text)
                predictions[model_name] = (emotion, stats)
            
            # Display results
            print_results(text, predictions)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
