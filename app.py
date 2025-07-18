import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import load_model, Sequential
import numpy as np
import pandas as pd
import json
import os
import gdown # Para baixar do Google Drive

# --- Funções de Carregamento (usando st.cache_resource para performance) ---

@st.cache_resource
def download_file_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        st.write(f"Baixando {output_path} do Google Drive...") # Debug: Avisa que está baixando
        try:
            gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)
            st.write(f"Download de {output_path} concluído.") # Debug: Confirma download
        except Exception as e:
            st.error(f"Erro ao baixar {output_path} do Google Drive: {e}")
            st.stop()
    else:
        st.write(f"{output_path} já existe localmente. Pulando download.") # Debug: Avisa que já existe

@st.cache_resource
def load_and_adapt_text_vectorizer(vectorizer_config_path):
    download_file_from_drive(VECTORIZER_DRIVE_ID, vectorizer_config_path)

    with open(vectorizer_config_path, 'r') as f:
        config_data = json.load(f)

    loaded_vectorizer = TextVectorization.from_config(config_data['config'])
    mock_data = tf.data.Dataset.from_tensor_slices(["a b c"]).batch(1)
    loaded_vectorizer.adapt(mock_data)
    loaded_vectorizer.set_weights(config_data['weights'])

    # --- DEBUG: Verifique o vocabulário carregado ---
    st.write("--- Debug: TextVectorizer ---")
    st.write(f"Tamanho do vocabulário carregado: {len(loaded_vectorizer.get_vocabulary())}")
    st.write(f"Primeiras 10 palavras do vocabulário: {loaded_vectorizer.get_vocabulary()[:10]}")
    st.write(f"Últimas 10 palavras do vocabulário: {loaded_vectorizer.get_vocabulary()[-10:]}")
    st.write("----------------------------")
    # --- Fim DEBUG ---

    return loaded_vectorizer

@st.cache_resource
def load_trained_news_model(model_path):
    download_file_from_drive(MODEL_DRIVE_ID, model_path)
    model = load_model(model_path, compile=False)
    # Recompile para garantir que o otimizador está configurado corretamente
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# --- Caminhos dos Arquivos Salvos (ajuste seus IDs do Drive aqui) ---
MODEL_FILE_NAME = 'fake_news_classifier.keras' # Ou 'fake_news_detector_model.h5'
VECTORIZER_CONFIG_FILE_NAME = 'text_vectorizer_config.json'

# **AQUI VOCÊ PRECISA COLOCAR OS IDs CORRETOS DOS SEUS ARQUIVOS NO GOOGLE DRIVE**
# Ex: se o link do seu modelo é https://drive.google.com/file/d/SEU_MODEL_ID/view
MODEL_DRIVE_ID = '1GpWix8dp6FeFAs6g0etbnw_avu9Aflfp' # <--- SUBSTITUA PELO ID DO SEU MODELO .keras OU .h5
VECTORIZER_DRIVE_ID = '1QxY2zR3w4E5t6Y7u8I9o0Pq1R2S3tU4V' # <--- SUBSTITUA PELO ID DO SEU ARQUIVO .json


# --- Função de Previsão para o Modelo de Notícias ---
def predict_news_sentiment(model_instance, text_vectorizer_instance, news_text):
    # --- DEBUG: Verifique o texto de entrada e a vetorização ---
    st.write("--- Debug: Entrada e Vetorização ---")
    st.write(f"Texto original de entrada: '{news_text}'")
    
    # Vetoriza o texto e converte para numpy para inspeção
    vectorized_text_tensor = text_vectorizer_instance(tf.constant([news_text]))
    vectorized_text_np = vectorized_text_tensor.numpy()

    st.write(f"Sequência vetorizada (parte inicial): {vectorized_text_np[0][:20]}")
    st.write(f"Sequência vetorizada (parte final): {vectorized_text_np[0][-20:]}")
    st.write(f"Shape da entrada para o modelo: {vectorized_text_tensor.shape}")
    st.write("------------------------------------")
    # --- Fim DEBUG ---

    prediction_prob = model_instance.predict(vectorized_text_tensor)[0][0]
    classes = ['Falsa', 'Real']
    probabilities = [ (1 - prediction_prob) * 100, prediction_prob * 100 ]
    df_results = pd.DataFrame({'Classes': classes, 'Probabilidades (%)': probabilities})
    predicted_class = "REAL" if prediction_prob > 0.5 else "FALSA"
    return predicted_class, prediction_prob, df_results

# --- Função Principal do Aplicativo Streamlit (main) ---
def main():
    st.set_page_config(page_title="Detector de Notícias Falsas")

    st.title('📰 Detector de Notícias Falsas')
    st.markdown("Use este aplicativo para verificar se uma notícia é provavelmente **FALSA** ou **REAL**.")

    st.info("Preparando o detector... Isso pode levar alguns segundos na primeira vez (baixando o modelo).")
    
    text_vectorizer = load_and_adapt_text_vectorizer(VECTORIZER_CONFIG_FILE_NAME)
    model = load_trained_news_model(MODEL_FILE_NAME)
    
    st.success("Detector pronto! Insira o texto da notícia.")

    st.write("---")
    st.subheader('Cole o texto da notícia aqui:')
    user_news_input = st.text_area("", height=250, placeholder="Ex: Cientistas descobrem cidade perdida em Marte...")

    if st.button('Classificar Notícia'):
        if user_news_input:
            with st.spinner('Analisando notícia...'):
                predicted_class, prediction_prob, df_results = predict_news_sentiment(model, text_vectorizer, user_news_input)

                st.subheader("Resultado da Análise:")
                if predicted_class == "REAL":
                    st.success(f"**Esta notícia é REAL!** (Probabilidade de ser Real: {prediction_prob*100:.2f}%)")
                    st.balloons()
                else:
                    st.error(f"**Esta notícia é FALSA!** (Probabilidade de ser Falsa: {(1-prediction_prob)*100:.2f}%)")
                    st.snow()

                st.write("---")
                st.subheader("Probabilidades Detalhadas:")
                st.dataframe(df_results.set_index('Classes'))

        else:
            st.warning('Por favor, digite ou cole o texto de uma notícia para classificar.')

    st.markdown("---")
    st.write("Desenvolvido com TensorFlow e Streamlit")
    st.write("Para treinar o modelo, execute `python model_trainer.py` primeiro e depois faça o upload dos arquivos `.keras` e `.json` para o Google Drive e atualize os IDs neste script.")

if __name__ == "__main__":
    main()
