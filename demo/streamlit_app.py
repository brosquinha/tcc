import math
import os
from collections import namedtuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    # IDE imports
    from demo.lstm_data import arq_data, confiability_data, tests_data
    from demo.plot_custom_confusion_matrix import plot_custom_confusion_matrix
    from demo.table_of_contents import ToC
except ImportError:
    # Streamlit runtime imports
    from lstm_data import arq_data, confiability_data, tests_data
    from plot_custom_confusion_matrix import plot_custom_confusion_matrix
    from table_of_contents import ToC

# Instantiate and initialize Table of Contents
toc = ToC()
st.sidebar.title("Conteúdo")
# Create table of contents on sidebar with placeholder
toc.placeholder(st.sidebar)

base_path = os.environ.get("STREAMLIT_BASE_PATH", os.path.dirname(__file__))

# Initialize the introductory texts and add Subtitles and Headers to Table of Contents
with open(f'{base_path}/demo.md', 'r') as f:
    for line in f.readlines():
        if line.startswith("###"):
            toc.subheader(line[3:])
        elif line.startswith("##"):
            toc.header(line[2:])
        else:
            line

images_base_path = f'{base_path}/images/'

# Sidebar Select Boxes to choose technique, dataset, and emotion to display
tech = st.sidebar.selectbox("Técnica", ["Naive Bayes", "Rede neural"])
dataset = st.sidebar.selectbox("Dataset", ["SemEval", "TEC"])
emotion = st.sidebar.selectbox("Emoção", ["Raiva", "Medo", "Alegria", "Tristeza"])

tech_formatted = 'naive-bayes' if tech == "Naive Bayes" else "lstm"

# Display all Confusion matrixes for each test
for action in ["Testes", "Confiabilidade", "Validação", "Análise final"]:
    toc.subheader(f"{action}")
    if action == 'Análise final':
        filename = f'{images_base_path}resultados/popular_tweets_{dataset.lower()}_{tech_formatted}.png'
    else:
        tech_path_formatted = 'naive_bayes' if tech == "Naive Bayes" else "lstm_cnn_conc"
        action_path_formatted = {
            'Testes': 'avaliacao',
            'Confiabilidade': 'confiabilidade',
            'Validação': 'validacao'
        }
        emotion_formatted = {
            "Raiva": "anger",
            "Medo": "fear",
            "Alegria": "joy",
            "Tristeza": "sadness"
        }
        sufix = '-lstm' if tech_formatted == 'lstm' else ''
        filename = f'{images_base_path}matrixes/{tech_path_formatted}/{action_path_formatted[action]}/'+\
            f'{emotion_formatted[emotion]}-{dataset.lower()}{sufix}.png'

    if tech == "Rede neural" and action in ['Testes', 'Confiabilidade']:
        if action == 'Testes':
            arq = st.selectbox("Arquitetura", list(tests_data.keys()))

            arq_filename = f'{images_base_path}/arquiteturas/{arq_data[arq]}'
            st.image(arq_filename, width=300)

            plt = plot_custom_confusion_matrix(tests_data[arq][dataset][emotion], emotion, dataset)
        else:
            plt = plot_custom_confusion_matrix(confiability_data[dataset][emotion], emotion, dataset)
        st.pyplot(plt)
        plt.close()
    else:
        try:
            st.image(filename, use_column_width=True)
        except FileNotFoundError:
            "No image, sorry ☹️"

# Update Table of Contents to display all possible app contents
st.sidebar.write(toc.generate())

