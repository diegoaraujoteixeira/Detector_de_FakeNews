import streamlit as st
import tensorflow as tf
# Importe Keras de tf_keras para compatibilidade com Keras 3
import keras as tf_keras # Adicione esta linha
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
# Use load_model de tf_keras
from keras.models import load_model # <--- MUDE PARA 'from keras.models import load_model' ou 'from tf_keras.models import load_model'
# Se você usou deploy_model.save("fake_news_classifier.keras")
# Você não precisa mais do Sequential, numpy, pandas, json, os, gdown aqui,
# porque o modelo já carrega tudo. Mantenha os imports de Streamlit e TensorFlow
# e as camadas que estão na arquitetura do modelo.
import numpy as np # Mantenha se precisar para tf.constant
import pandas as pd # Mantenha para df_results
import json # Mantenha caso queira o vetorizador separado (menos ideal)
import os # Mantenha para verificar existencia de arquivo
import gdown # Mantenha para baixar do Google Drive


# --- Caminhos dos Arquivos Salvos (atualize seus IDs do Drive aqui) ---
MODEL_FILE_NAME = 'fake_news_classifier.keras' # <--- O NOME DO SEU NOVO ARQUIVO!
# Se você está salvando o modelo COMPLETO (com TextVectorization) no formato .keras,
# VOCÊ NÃO PRECISA MAIS DO ARQUIVO text_vectorizer_config.json SEPARADO.
# Então, você pode COMENTAR/REMOVER VECTORIZER_CONFIG_FILE_NAME e VECTORIZER_DRIVE_ID.
# VECTORIZER_CONFIG_FILE_NAME = 'text_vectorizer_config.json'
# VECTORIZER_DRIVE_ID = 'SEU_ID_DO_VECTORIZER_AQUI'

# **AQUI VOCÊ PRECISA COLOCAR OS IDs CORRETOS DO SEU NOVO ARQUIVO .keras NO GOOGLE DRIVE**
MODEL_DRIVE_ID = 'SEU_NOVO_ID_DO_MODELO_KERAS_AQUI' # <--- NOVO ID DO fake_news_classifier.keras

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
def load_complete_model(model_path): # Renomeei para indicar que carrega o modelo completo
    download_file_from_drive(MODEL_DRIVE_ID, model_path)
    # Carrega o modelo que já inclui a camada TextVectorization
    # Use o load_model do 'keras' ou 'tf_keras' diretamente para Keras 3
    model = load_model(model_path, compile=False) # compile=False ainda é uma boa prática
    # O modelo já vem com a camada TextVectorization configurada internamente.
    # Não precisa mais de recompilação do otimizador aqui, pois o modelo completo já foi salvo compilado.
    return model

# Remova a função load_and_adapt_text_vectorizer, pois ela não é mais necessária

# --- Função Principal do Aplicativo Streamlit ---
def main():
    st.set_page_config(page_title="Detector de Notícias Falsas")
    st.title('📰 Detector de Notícias Falsas')
    st.markdown("Use este aplicativo para verificar se uma notícia é provavelmente **FALSA** ou **REAL**.")

    st.info("Preparando o detector... Isso pode levar alguns segundos na primeira vez (baixando o modelo).")

    # Carrega o modelo completo. Ele já tem o vetorizador integrado.
    full_model = load_complete_model(MODEL_FILE_NAME) # <--- AQUI CARREGA O MODELO COMPLETO

    st.success("Detector pronto! Insira o texto da notícia.")

    st.write("---")
    st.subheader('Cole o texto da notícia aqui:')
    user_news_input = st.text_area("", height=250, placeholder="Ex: Cientistas descobrem cidade perdida em Marte...")

    if st.button('Classificar Notícia'):
        if user_news_input:
            with st.spinner('Analisando notícia...'):
                # A previsão agora é feita diretamente no modelo completo
                # O modelo completo espera um tensor de strings
                prediction_prob = full_model.predict(tf.constant([user_news_input]))[0][0]

                # ... (resto da lógica de exibição permanece o mesmo) ...

                st.subheader("Resultado da Análise:")
                if prediction_prob > 0.5:
                    st.success(f"**Esta notícia é REAL!** (Probabilidade de ser Real: {prediction_prob*100:.2f}%)")
                    st.balloons()
                else:
                    st.error(f"**Esta notícia é FALSA!** (Probabilidade de ser Falsa: {(1-prediction_prob)*100:.2f}%)")
                    st.snow()

                st.write("---")
                st.subheader("Probabilidades Detalhadas:")
                classes = ['Falsa', 'Real'] # Defina as classes aqui, pois não vêm do df_results
                probabilities = [(1 - prediction_prob) * 100, prediction_prob * 100]
                df_results = pd.DataFrame({'Classes': classes, 'Probabilidades (%)': probabilities})
                st.dataframe(df_results.set_index('Classes'))

        else:
            st.warning('Por favor, digite ou cole o texto de uma notícia para classificar.')

    st.markdown("---")
    st.write("Desenvolvido com TensorFlow e Streamlit")
    st.write("Para treinar o modelo, execute `python model_trainer.py` e faça o upload do arquivo `fake_news_classifier.keras` para o Google Drive e atualize o ID neste script.")

if __name__ == "__main__":
    main()
