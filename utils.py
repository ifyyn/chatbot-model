import os
import random
import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model

# ❌ Disable GPU untuk Railway
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ✅ Setup NLTK path
nltk_data_path = './nltk_data'
if not os.path.exists(f"{nltk_data_path}/tokenizers/punkt"):
    nltk.download('punkt', download_dir=nltk_data_path)
nltk.data.path.append(nltk_data_path)

lemmatizer = WordNetLemmatizer()

# ✅ Load model dan assets
try:
    model = load_model('chatbot_model.h5')
except Exception as e:
    print(f"[❌] Gagal load model: {e}")
    model = None

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.load(open('intents.json'))

ignore_chars = ['?', '!', '.', ',']

def clean_text(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens if w not in ignore_chars]
    return tokens

def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for i, word in enumerate(vocab):
            if word == w:
                bow[i] = 1
    return np.array(bow)

def predict_class(text):
    try:
        if model is None:
            raise Exception("Model belum tersedia.")
        bow = bag_of_words(text, words)
        res = model.predict(np.array([bow]))[0]

        threshold = 0.2
        results = [[i, r] for i, r in enumerate(res) if r > threshold]
        results.sort(key=lambda x: x[1], reverse=True)

        if results:
            intent = classes[results[0][0]]
            return [{'intent': intent, 'probability': str(results[0][1])}]
        else:
            return [{'intent': 'notfound', 'probability': '0'}]
    except Exception as e:
        print(f"[❌] Error predict_class: {e}")
        return [{'intent': 'notfound', 'probability': '0'}]

def get_response(intent_list):
    tag = intent_list[0]['intent']
    for intent in intents['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            if isinstance(response, str):
                response = {"text": response}
            return response, tag

    return {
        "title": "🤖 Maklum AI Baru",
        "text": "Kata yang Anda ketik belum ada di data saya. Tapi saya masih bisa bantu soal rute, lokasi, aktivitas, harga tiket, dan fasilitas wisata di Jerowaru 😊"
    }, "notfound"
