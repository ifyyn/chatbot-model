# NLP Chatbot - Tahapan sesuai Metode Penelitian
import json
import random
import pickle
import numpy as np
import nltk
import matplotlib.pyplot as plt
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Data Collection dan Text Collection
# ----------------------------------
# Load data dari file intents.json yang berisi dataset chatbot wisata
with open('intents.json') as f:
    data = json.load(f)

# Text Cleaning dan Preprocessing
# -------------------------------
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_chars = ['?', '!', '.', ',']

for intent in data['intents']:
    for pattern in intent['patterns']:
        tokens = word_tokenize(pattern)
        words.extend(tokens)
        documents.append((tokens, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_chars]
words = sorted(set(words))
classes = sorted(set(classes))

# Simpan kamus kata dan kelas
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Feature Engineering
# -------------------
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
    for w in words:
        bag.append(1 if w in pattern_words else 0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Modeling
# --------
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(train_y[0]), activation='softmax'))

opt = Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Model Training
history = model.fit(train_x, train_y, epochs=300, verbose=1)
model.save('chatbot_model.h5')

# Evaluation
# ----------
pd.DataFrame(history.history).to_csv('training_history.csv', index=False)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Akurasi', color='blue')
plt.title('Akurasi Model Selama Training')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss', color='red')
plt.title('Loss Model Selama Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Monitoring
# ----------
print("\n\U0001F4CB Dataset (patterns dan tag):\n")
df = pd.DataFrame([(pattern, intent['tag']) for intent in data['intents'] for pattern in intent['patterns']],
                  columns=['Pattern', 'Tag'])
print(df.head(6))
