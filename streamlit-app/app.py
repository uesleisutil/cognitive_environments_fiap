import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

# Função para carregar os componentes necessários
def load_components(model_path, tokenizer_path, label_encoder_path):
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(label_encoder_path, 'rb') as enc:
        label_encoder = pickle.load(enc)
    return model, tokenizer, label_encoder

# Função de normalização do texto
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
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Função para fazer predições
def make_prediction(model, tokenizer, label_encoder, text):
    max_length = 200  # Deve corresponder ao valor usado durante o treinamento
    text = normalize_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded_sequence)
    pred_index = np.argmax(prediction, axis=1)
    pred_label = label_encoder.inverse_transform(pred_index)[0]
    return pred_label

# Streamlit app
def main():
    st.title('Classificador de Reclamações')
    model_path = '../model/model.h5'
    tokenizer_path = '../model/tokenizer.pkl'
    label_encoder_path = '../model/label_encoder.pkl'
    
    model, tokenizer, label_encoder = load_components(model_path, tokenizer_path, label_encoder_path)
    
    user_input = st.text_area("Digite sua reclamação aqui:")
    
    if st.button('Classificar'):
        if user_input:
            prediction = make_prediction(model, tokenizer, label_encoder, user_input)
            st.write(f'Categoria prevista: {prediction}')
        else:
            st.write("Por favor, insira uma reclamação para classificar.")

if __name__ == '__main__':
    main()