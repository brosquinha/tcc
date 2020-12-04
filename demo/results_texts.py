results_texts = {
    "Testes": """
    A matriz de confusão acima ilustra o resultado de testes realizados com o classificador
    de {emotion} com uma parcela do próprio _dataset_ {dataset}.
    """,
    
    "Confiabilidade": """
    A matriz de confusão acima ilustra o resultado de testes de confiabilidade com a totalidade do _dataset_ {other_dataset}
    com o classificador de {emotion} treinado com o _dataset_ {dataset}.
    """,
    
    "Validação": """
    A matriz de confusão acima ilustra o resultado de testes de validação 
    com o classificador de {emotion} treinado com o _dataset_ {dataset} nos _tweets_ classificados a mão.
    """,
    
    "Arquitetura": """
    Figura ilustrativa de como os neurônios da arquitetura selecionada se relacionam.
    """,

    "Análise final": """
    Para os classificadores {tech}, utilizou-se cada conjunto de classificadores treinados em
    um _dataset_ específico para fazer a análise das respostas, obtendo um gráfico para cada conjunto.
    """,
}