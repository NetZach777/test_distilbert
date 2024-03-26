import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf

app = FastAPI()

# Tokenizer Hugging Face
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Modèle entraîné
model = TFDistilBertForSequenceClassification.from_pretrained('models')

class TweetRequest(BaseModel):
    text: str

@app.post("/predict-sentiment/")
def predict_sentiment(tweet: TweetRequest):
    # Vérifie si le texte est vide ou ne contient que des espaces blancs
    if not tweet.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty or just whitespace.")

    inputs = tokenizer(tweet.text, return_tensors="tf", max_length=512, truncation=True, padding="max_length")
    outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    prediction = tf.argmax(outputs.logits, -1).numpy()[0]
    # S'assurez que le modèle renvoie des prédictions binaires (0 ou 1)
    if prediction not in [0, 1]:
        raise HTTPException(status_code=500, detail="Model prediction was not 0 or 1.")
    sentiment = "positive" if prediction == 1 else "negative"
    return {"sentiment": sentiment}