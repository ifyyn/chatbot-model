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
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [classes[r[0]] for r in results]

def get_response(intent_list):
    if not intent_list:
        return "Maaf, saya tidak mengerti pertanyaan Anda."
    tag = intent_list[0]
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Maaf, saya tidak mengerti pertanyaan Anda."
