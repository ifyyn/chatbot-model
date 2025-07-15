import random
import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model

nltk.download('punkt')
lemmatizer = WordNetLemmatizer()

# Load data
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.load(open('intents.json'))
model = load_model('chatbot_model.h5')

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
    bow = bag_of_words(text, words)
    res = model.predict(np.array([bow]))[0]
    threshold = 0.2
    results = [[i, r] for i, r in enumerate(res)]
    results.sort(key=lambda x: x[1], reverse=True)

    if results and results[0][1] >= threshold:
        return [{'intent': classes[results[0][0]], 'probability': str(results[0][1])}]
    else:
        return [{'intent': 'notfound', 'probability': '0'}]

def get_response(intent_list):
    tag = intent_list[0]['intent']
    for intent in intents['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            if isinstance(response, str):
                response = {"text": response}  # bungkus string dalam objek
            return response, tag
    # fallback
    return {
        "title": "🤖 Maklum AI Baru",
        "text": "Kata yang Anda ketik belum ada di data saya. Maklum, saya AI baru. Tapi saya masih bisa bantu soal rute, lokasi, aktivitas, harga tiket, dan fasilitas di Jerowaru 😊"
    }, "notfound"


def chatbot_response(user_input):
    intents_detected = predict_class(user_input)
    response, context = get_response(intents_detected)
    return response, context
