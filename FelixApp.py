# -*- coding: utf-8 -*-
# ========== 🎨 CONFIGURAÇÃO INICIAL (DEVE SER A PRIMEIRA CHAMADA) ==========
import streamlit as st
st.set_page_config(
    page_title="Diagnóstico Multimoda de Equipamentos Com Risco de Falhas Utilizandos LLMS em Refinarias e Plataformas de Petroleo",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== 📦 IMPORTAÇÕES ==========
import pandas as pd
import requests
import json
from PIL import Image
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import time

# ========== 🔐 CONFIGURAÇÃO DA API DEEPSEEK ==========
DEFAULT_API_KEY = "sk-19ed188057ff4b49b07450a97fbabdd8"  # Substitua pela sua chave

# Tenta carregar do secrets.toml, senão usa padrão ou input do usuário
try:
    api_key = st.secrets["deepseek_api_key"]
except (FileNotFoundError, AttributeError, KeyError):
    api_key = DEFAULT_API_KEY
    if not api_key:
        api_key = st.text_input("🔑 Insira sua DeepSeek API Key", type="password")
        if not api_key:
            st.error("❌ API Key é obrigatória para continuar.")
            st.stop()

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"  # Verifique o endpoint atual
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# ========== 🖥️ INTERFACE DO USUÁRIO ==========
st.title("⚙️ Diagnóstico Multimodal de Equipamentos Com Risco de Falhas Utilizandos LLMS em Refinarias e Plataformas de Petroleo")
st.markdown("""
    **Analise imagens, áudios e dados técnicos** para identificar falhas potenciais em equipamentos 
    utilizando LLMS**.
""")

# ========== 📝 FORMULÁRIO DE ENTRADA ==========
with st.expander("📤 Carregar Dados do Equipamento", expanded=True):
    tab1, tab2, tab3 = st.tabs(["📷 Imagem", "🎙️ Áudio", "📊 Dados Técnicos"])
    
    # TAB 1: Upload de Imagem
    with tab1:
        img_file = st.file_uploader("Imagem do equipamento (JPG/PNG)", type=["jpg", "png", "jpeg"])
        if img_file:
            image = Image.open(img_file)
            st.image(image, caption="Imagem carregada", use_column_width=True)
    
    # TAB 2: Upload de Áudio
    with tab2:
        audio_file = st.file_uploader("Gravação de áudio (WAV/MP3)", type=["wav", "mp3"])
        if audio_file:
            st.audio(audio_file)
            try:
                y, sr = librosa.load(audio_file, sr=None)
                fig, ax = plt.subplots(figsize=(10, 4))
                D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
                img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
                fig.colorbar(img, ax=ax, format="%+2.0f dB")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Erro ao processar áudio: {str(e)}")
    
    # TAB 3: Upload de CSV
    with tab3:
        csv_file = st.file_uploader("Dados técnicos (CSV)", type="csv")
        df = None
        if csv_file:
            try:
                df = pd.read_csv(csv_file)
                st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Erro ao ler CSV: {str(e)}")
    
    # Campo de observações
    observations = st.text_area(
        "📝 Observações do Operador",
        placeholder="Descreva sintomas observados (ruídos, vibrações, falhas...)",
        height=150
    )

# ========== 🧠 ANÁLISE COM DEEPSEEK ==========
if st.button("🔍 Executar Análise Completa", type="primary", use_container_width=True):
    # Verificação segura dos dados de entrada
    if df is None or df.empty:
        st.error("❌ Por favor, carregue um arquivo CSV válido.")
        st.stop()
    
    if not observations.strip():
        st.error("❌ Descreva as observações do operador.")
        st.stop()

    progress_bar = st.progress(0, text="Preparando análise...")
    
    try:
        # Prepara os dados
        csv_data = df.to_csv(index=False)
        progress_bar.progress(20, text="Processando dados...")
        
        # Construção do prompt
        prompt = f"""
        Como especialista em manutenção preditiva, analise os seguintes dados:

        [DADOS TÉCNICOS DO EQUIPAMENTO]
        {csv_data}

        [OBSERVAÇÕES DO OPERADOR]
        {observations}

        Forneça um relatório estruturado com:
        1. 🔍 Diagnóstico Técnico
        2. ⚠️ Nível de Risco (Baixo/Médio/Alto/Crítico)
        3. 🛠️ Ações Recomendadas (curto, médio e longo prazo)
        4. 📅 Cronograma de Manutenção Sugerido
        5. 💡 Recomendações Adicionais

        Seja técnico, mas claro. Use **marcação** para organização.
        """
        
        progress_bar.progress(40, text="Conectando ao DeepSeek...")
        
        # Chamada à API
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "Você é um engenheiro especialista em manutenção industrial."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1500
        }
        
        with st.spinner("🔎 Analisando com DeepSeek AI..."):
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            progress_bar.progress(80, text="Processando resultados...")
            
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                reply = result['choices'][0]['message']['content']
                progress_bar.progress(100, text="✅ Análise concluída!")
                time.sleep(0.5)
                progress_bar.empty()
                
                st.success("## 📋 Relatório de Diagnóstico")
                st.markdown(reply)
                
                # Botão de download
                st.download_button(
                    label="📥 Baixar Relatório",
                    data=reply,
                    file_name=f"diagnostico_equipamento_{time.strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
            else:
                st.error("❌ Resposta inesperada da API.")
                st.json(result)  # Debug
    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Falha na comunicação com a API: {str(e)}")
    except Exception as e:
        st.error(f"❌ Erro inesperado: {str(e)}")

# ========== 🚨 SEÇÃO DE DEBUG (OPCIONAL) ==========
if st.sidebar.checkbox("Mostrar informações técnicas"):
    st.sidebar.subheader("🔧 Debug")
    st.sidebar.write(f"🔑 API Key: {'✅ Configurada' if api_key else '❌ Não configurada'}")
    if api_key:
        st.sidebar.code(f"Últimos 5 chars: {api_key[-5:]}")
    st.sidebar.write("📊 Dados carregados:", f"{len(df)} linhas" if df is not None else "Nenhum")