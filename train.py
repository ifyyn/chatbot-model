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

# 1. Data Collection
with open('intents.json') as f:
    data = json.load(f)

# 2. Preprocessing & Text Cleaning
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_chars = ['?', '!', '.', ',']

# 2.2 Pembersihan Teks
print("\n📚 Output 2.2 - Contoh Hasil Pembersihan Teks:")
sample_clean = []

for intent in data['intents']:
    if 'patterns' in intent and isinstance(intent['patterns'], list):
        for pattern in intent['patterns'][:2]:
            if pattern.strip():  # lewati jika kosong
                tokens_awal = word_tokenize(pattern)
                tokens_lower = [w.lower() for w in tokens_awal if w not in ignore_chars]
                tokens_lemma = [lemmatizer.lemmatize(w) for w in tokens_lower]
                sample_clean.append({
                    "Original": pattern,
                    "Tokenisasi": tokens_awal,
                    "Lowercase": tokens_lower,
                    "Lemmatized": tokens_lemma,
                    "Tag": intent['tag']
                })

for i, row in enumerate(sample_clean[:5]):
    print(f"\n{i+1}. Tag: {row['Tag']}")
    print(f"   Kalimat Asli     : {row['Original']}")
    print(f"   Tokenisasi       : {row['Tokenisasi']}")
    print(f"   Lowercase        : {row['Lowercase']}")
    print(f"   Setelah Lemmatize: {row['Lemmatized']}")

# Tambahan: Cetak proses lemmatization detail
print("\n📌 Output Tambahan - Proses Lemmatization (Detail per pattern):\n")
for intent in data['intents']:
    if 'patterns' in intent and isinstance(intent['patterns'], list):
        for pattern in intent['patterns'][:2]:
            if pattern.strip():
                print("Sebelum tokenizing      :", pattern)
                tokens = word_tokenize(pattern)
                print("Setelah tokenizing      :", tokens)
                lowercased = [w.lower() for w in tokens if w not in ignore_chars]
                print("Setelah lowercase       :", lowercased)
                lemmatized = [lemmatizer.lemmatize(w) for w in lowercased]
                print("Setelah lemmatization   :", lemmatized)
                print("Tag intent              :", intent['tag'])
                print("-" * 70)

# Lanjutkan preprocessing
for intent in data['intents']:
    if 'patterns' in intent and isinstance(intent['patterns'], list):
        for pattern in intent['patterns']:
            if pattern.strip():
                tokens = word_tokenize(pattern)
                documents.append((tokens, intent['tag']))
                words.extend(tokens)
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

# Bersihkan kata
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_chars]
words = sorted(set(words))
classes = sorted(set(classes))

print("\n📌 Hasil Preprocessing:")
print(f"Jumlah kata unik: {len(words)}")
print(f"Kata-kata unik: {words[:20]} ...")
print(f"Jumlah intent: {len(classes)}")
print(f"Daftar intent: {classes}")

print(f"\nContoh dokumen:")
for i, doc in enumerate(documents[:5]):
    print(f"{i+1}. Tokens: {doc[0]} | Tag: {doc[1]}")

# Simpan kamus
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# 2.3 Feature Engineering
print("\n🧠 Output 2.3 - Contoh Ekstraksi Fitur (Pattern, BoW, One-hot):")
training = []
output_empty = [0] * len(classes)
detailed_features = []

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0] if w not in ignore_chars]
    for w in words:
        bag.append(1 if w in pattern_words else 0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

    pattern_text = ' '.join(doc[0]) if len(doc[0]) > 0 else '(pattern kosong)'
    detailed_features.append({
        "pattern": pattern_text,
        "bag_of_words": bag,
        "output_class": output_row
    })

# Acak dan ubah ke array
random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Tambahan: Cetak 5 contoh BoW
print("\n🔍 Output Tambahan - Bag of Words (5 Contoh Pattern):")
for i, doc in enumerate(documents[:5]):
    original_pattern = ' '.join(doc[0])
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0] if w not in ignore_chars]
    bow_vector = [1 if w in pattern_words else 0 for w in words]
    print(f"\n{i+1}. Pattern Asli   : {original_pattern}")
    print(f"    Tokenisasi     : {doc[0]}")
    print(f"    Setelah Lemma  : {pattern_words}")
    print(f"    Tag Intent     : {doc[1]}")
    print(f"    Bag of Words   : {bow_vector}")

# Tampilkan 3 hasil detail Feature Engineering
for i in range(3):
    item = detailed_features[i]
    print(f"\nContoh {i+1}")
    print(f"Pattern             : {item['pattern']}")
    print(f"Binary Representation (BoW): {item['bag_of_words']}")
    print(f"Output Class (One-hot)     : {item['output_class']}")

# Modeling
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
opt = Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Training model
history = model.fit(train_x, train_y, epochs=200, verbose=1)
model.save('chatbot_model.h5')

# Evaluation
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

# Monitoring Dataset
print("\n🗂️ Cuplikan Dataset (Pattern & Tag):")
df = pd.DataFrame(
    [(pattern, intent['tag']) for intent in data['intents'] 
     if 'patterns' in intent and isinstance(intent['patterns'], list)
     for pattern in intent['patterns'] if pattern.strip()],
    columns=['Pattern', 'Tag']
)
print(df.head(10))
