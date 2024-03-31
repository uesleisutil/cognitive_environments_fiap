import pandas as pd
import re
import boto3
import string
import spacy
import nltk
import os
import unicodedata
import pickle
from numpy import argmax
from spacy.lang.pt.stop_words import STOP_WORDS
from io import StringIO
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from dotenv import load_dotenv
from tensorflow.keras.callbacks import EarlyStopping

# Load variables from .env
load_dotenv()

# Access environment variables.
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
bucket_name = os.getenv('BUCKET_NAME')
object_key = os.getenv('OBJECT_KEY')

# Try to access AWS, if not, then load local data.
try:
    s3_client = boto3.client('s3', region_name='us-east-1', 
                         aws_access_key_id=aws_access_key_id, 
                         aws_secret_access_key=aws_secret_access_key)
    csv_obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_string),sep=';')
    print("Loaded data from S3.")
except Exception as e:
    print(f"Failed to load data from S3 due to {e}. Attempting to load from local directory.")
    local_path = '../data/tickets_reclamacoes_classificados.csv'
    df = pd.read_csv(local_path,sep=';')
    print("Loaded data from local directory.")
    
# Portuguese data for Space.
nlp = spacy.load("pt_core_news_sm")

# Functions to normalize text
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-záéíóúàèìòùâêîôûãõäëïöüç\s]', '', text)
    text = ''.join((c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn'))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize_and_remove_stopwords(text):
    """
    Apply lemmatization to the input string and remove stopwords using the
    spaCy library for the Portuguese language. Only non-stopword tokens are
    lemmatized and concatenated into a single string.
    
    Parameters:
    text (str): The input string to be lemmatized and from which stopwords will be removed.
    
    Returns:
    str: The lemmatized string with stopwords removed.
    """
    doc = nlp(text)
    result = []
    for token in doc:
        if not token.is_stop:
            result.append(token.lemma_)
    return ' '.join(result)

# Normalize and process data in 'descricao_reclamacao' and 'categoria' columns
for col in ['descricao_reclamacao', 'categoria']:
    df[f'{col}_norm'] = df[col].apply(normalize_text).apply(lemmatize_and_remove_stopwords)

# Splitting data into training and testing sets
X = df['descricao_reclamacao_norm']
y = df['categoria_norm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Tokenizing text
max_words = 20000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding sequences
max_length = 200
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# Encoding labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
y_train_encoded = to_categorical(y_train_encoded)
y_test_encoded = to_categorical(y_test_encoded)

# Model architecture
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=100, input_length=max_length))
model.add(Bidirectional(LSTM(100, return_sequences=False, dropout=0.5, recurrent_dropout=0.5, kernel_regularizer=l2(0.01))))
model.add(Dense(y_train_encoded.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_accuracy', patience=2)

# Model training
model.fit(X_train_pad, y_train_encoded, epochs=16, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

# Prediction and evaluation
y_pred = model.predict(X_test_pad)
y_pred_classes = argmax(y_pred, axis=1)
y_test_classes = argmax(y_test_encoded, axis=1)
print(classification_report(y_test_classes, y_pred_classes, target_names=label_encoder.classes_))

# Save the model, tokenizer, and label encoder to files
model.save('model.h5')
with open('tokenizer.pkl', 'wb') as tkn_file:
    pickle.dump(tokenizer, tkn_file)
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)