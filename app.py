import streamlit as st
import tensorflow as tf

import keras as tf_keras # Adicione esta linha
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization

from keras.models import load_model # <--- MUDE PARA 'from keras.models import load_model' ou 'from tf_keras.models import load_model'

import numpy as np # Mantenha se precisar para tf.constant
import pandas as pd # Mantenha para df_results
import json # Mantenha caso queira o vetorizador separado (menos ideal)
import os # Mantenha para verificar existencia de arquivo
import gdown # Mantenha para baixar do Google Drive


# --- Caminhos dos Arquivos Salvos---
MODEL_FILE_NAME = 'fake_news_classifier.keras' 

MODEL_DRIVE_ID = 'SEU_NOVO_ID_DO_MODELO_KERAS_AQUI' 

# --- Funções de Carregamento (simplificadas para o formato .keras) ---

@st.cache_resource
def download_file_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        st.write(f"Baixando {output_path} do Google Drive...")
        try:
            gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)
            st.write(f"Download de {output_path} concluído.")
        except Exception as e:
            st.error(f"Erro ao baixar {output_path} do Google Drive: {e}")
            st.stop()
    else:
        st.write(f"{output_path} já existe localmente. Pulando download.")

@st.cache_resource
def load_complete_model(model_path):
    download_file_from_drive(MODEL_DRIVE_ID, model_path)
    
    model = load_model(model_path, compile=False) # compile=False ainda é uma boa prática
    
    return model

# --- Função Principal do Aplicativo Streamlit ---
def main():
    st.set_page_config(page_title="Detector de Notícias Falsas")
    st.title('📰 Detector de Notícias Falsas')
    st.markdown("Use este aplicativo para verificar se uma notícia é provavelmente **FALSA** ou **REAL**.")

    st.info("Preparando o detector... Isso pode levar alguns segundos na primeira vez (baixando o modelo).")

    # Carrega o modelo completo
    full_model = load_complete_model(MODEL_FILE_NAME)

    st.success("Detector pronto! Insira o texto da notícia.")

    st.write("---")
    st.subheader('Cole o texto da notícia aqui:')
    user_news_input = st.text_area("", height=250, placeholder="Ex: Cientistas descobrem cidade perdida em Marte...")

    if st.button('Classificar Notícia'):
        if user_news_input:
            with st.spinner('Analisando notícia...'):
                
                prediction_prob = full_model.predict(tf.constant([user_news_input]))[0][0]

                st.subheader("Resultado da Análise:")
                if prediction_prob > 0.5:
                    st.success(f"**Esta notícia é REAL!** (Probabilidade de ser Real: {prediction_prob*100:.2f}%)")
                    st.balloons()
                else:
                    st.error(f"**Esta notícia é FALSA!** (Probabilidade de ser Falsa: {(1-prediction_prob)*100:.2f}%)")
                    st.snow()

                st.write("---")
                st.subheader("Probabilidades Detalhadas:")
                classes = ['Falsa', 'Real'] 
                probabilities = [(1 - prediction_prob) * 100, prediction_prob * 100]
                df_results = pd.DataFrame({'Classes': classes, 'Probabilidades (%)': probabilities})
                st.dataframe(df_results.set_index('Classes'))

        else:
            st.warning('Por favor, digite ou cole o texto de uma notícia para classificar.')

    

if __name__ == "__main__":
    main()
