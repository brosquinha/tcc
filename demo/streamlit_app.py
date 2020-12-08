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
    from demo.results_texts import results_texts
    from demo.table_of_contents import ToC
except ImportError:
    # Streamlit runtime imports
    from lstm_data import arq_data, confiability_data, tests_data
    from plot_custom_confusion_matrix import plot_custom_confusion_matrix
    from results_texts import results_texts
    from table_of_contents import ToC

# Instantiate and initialize Table of Contents
toc = ToC()
st.sidebar.title("Conteúdo")
# Create table of contents on sidebar with placeholder
toc.placeholder(st.sidebar)

base_path = os.environ.get("STREAMLIT_BASE_PATH", os.path.dirname(__file__))

with open(f"{base_path}/style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize the introductory texts and add Subtitles and Headers to Table of Contents
with open(f'{base_path}/demo.md', 'r') as f:
    for line in f.readlines():
        if line.startswith("###"):
            toc.subheader(line[3:])
        elif line.startswith("##"):
            toc.header(line[2:])
        else:
            line

images_base_path = f'{base_path}/images'

# Sidebar Select Boxes to choose technique, dataset, and emotion to display
tech = st.sidebar.selectbox("Técnica", ["Naive Bayes", "Rede neural"])
dataset = st.sidebar.selectbox("Dataset", ["SemEval", "TEC"])
emotion = st.sidebar.selectbox("Emoção", ["Raiva", "Medo", "Alegria", "Tristeza"])

tech_formatted = 'naive-bayes' if tech == "Naive Bayes" else "lstm"

# Display all Confusion matrixes for each test
for action in ["Testes", "Confiabilidade", "Validação", "Análise final"]:
    comment = results_texts[action].format(
        emotion=emotion.lower(),
        dataset=dataset,
        other_dataset='TEC' if dataset == 'SemEval' else 'SemEval',
        tech=tech.lower()
    )
    toc.subheader(f"{action}")
    if action == 'Análise final':
        filename = f'{images_base_path}/resultados/popular_tweets_{dataset.lower()}_{tech_formatted}.png'
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
        filename = f'{images_base_path}/matrixes/{tech_path_formatted}/{action_path_formatted[action]}/'+\
            f'{emotion_formatted[emotion]}-{dataset.lower()}{sufix}.png'

    if tech == "Rede neural" and action in ['Testes', 'Confiabilidade']:
        matrix_filename = f'{images_base_path}/matrixes/'
        if action == 'Testes':
            arq = st.selectbox("Arquitetura", list(tests_data.keys()))

            arq_filename = f'{images_base_path}/arquiteturas/{arq_data[arq]}.png'
            matrix_filename += f'{arq_data[arq]}/avaliacao/'
            matrix_filename += '{emotion_name}-{dataset_name}.png'
            st.image(arq_filename)

            plt_title = f'Confusion matrix for {emotion.lower()} with {arq_data[arq]} ({dataset} dataset)'
            st.write(results_texts['Arquitetura'])

            plt = plot_custom_confusion_matrix(
                tests_data[arq][dataset][emotion], emotion, dataset, None,
                plt_title)
        else:
            matrix_filename += 'lstm_cnn_conc/confiabilidade/{emotion_name}-{dataset_name}.png'
            other_dataset = 'SemEval' if dataset == 'TEC' else 'TEC'
            plt_title = f'Confusion matrix for {emotion.lower()} (trained on {dataset} and tested on {other_dataset})'
            plt = plot_custom_confusion_matrix(
                confiability_data[dataset][emotion], emotion, dataset, None,
                plt_title)
        st.pyplot(plt)
        plt.close()
    else:
        try:
            st.image(filename, use_column_width=True)
        except FileNotFoundError:
            "No image, sorry ☹️"
    st.write(comment)

toc.header("Conclusões")
"""
Depois de treinados e validados os classificadores desenvolvidos pelo projeto,
observou-se que aqueles desenvolvidos com Naive Bayes demonstraram melhores métricas
de classificação de emoções durante os testes de confiabilidade
e validação. Assim, dando preferência aos resultados obtidos na seção
anterior por esses classificadores, o que observou-se é que, para
_tweets_ mais populares da base, a porcentagem de respostas alegres é consideravelmente menor que as demais emoções. 
"""

st.image(
    f'{images_base_path}/resultados/popular_tweets_aggregate_naive-bayes.png',
    use_column_width=True,
    caption="""
    Gráfico de dispersão ilustrando a popularidade de tweets propagadores
    de fake news e a relação com a porcentagem de respostas classificadas
    com as quatro emoções. Também estão ilustradas as linhas de tendências de cada série.
    As respostas foram classificadas utilizando os classificadores Naive Bayes treinados
    com os datasets SemEval e TEC. Pontos outliers foram removidos.
    """
)

"""
Com o desenvolvimento deste projeto, foram desenvolvidos _scripts_ coletores de _tweets_, _retweets_ 
e respostas de forma automática e em massa, possibilitando que essa grande quantidade de informações sobre essa 
relevante mídia social possa ser analisado para as mais diversas pesquisas sobre processamento de linguagem natural.

Outro ponto de contribuição do projeto foi o desenvolvimento de classificadores de emoções individuais, em que 
um pequeno texto pode ser testado em múltiplas emoções, ao invés de se determinar apenas uma emoção predominante. 
Embora esses modelos ainda necessitem de mais trabalho para alcançarem métricas de desempenho mais satisfatórias, 
é um bom ponto de partida para futuras empreitadas.

Por fim, a maior contribuição desse projeto é a relação apontada entre emoções provocadas por notícias falsas e 
sua popularidade. Observou-se que as notícias falsas mais populares provocam consideravelmente mais tristeza e 
raiva do que alegria, o que pode ser alvo de futuras pesquisas a cerca dos motivos pelos quais esse 
relacionamento existe.
"""

# Update Table of Contents to display all possible app contents
st.sidebar.write(toc.generate())

