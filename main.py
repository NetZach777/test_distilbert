import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf

# Désactiver l'utilisation du GPU si non disponible
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI()

# Charger le tokenizer de Hugging Face
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Charger le modèle entraîné
try:
    model = TFDistilBertForSequenceClassification.from_pretrained('models')
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading the model: {str(e)}")

class TweetRequest(BaseModel):
    text: str

@app.post("/predict-sentiment/")
def predict_sentiment(tweet: TweetRequest):
    # Vérification que le texte n'est pas vide ou constitué d'espaces blancs
    if not tweet.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty or just whitespace.")

    # Tokeniser l'entrée
    try:
        inputs = tokenizer(tweet.text, return_tensors="tf", max_length=512, truncation=True, padding="max_length")
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        prediction = tf.argmax(outputs.logits, -1).numpy()[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    # Vérification que la prédiction est binaire (0 ou 1)
    if prediction not in [0, 1]:
        raise HTTPException(status_code=500, detail="Model prediction was not 0 or 1.")
    
    sentiment = "positive" if prediction == 1 else "negative"
    
    return {"sentiment": sentiment}
