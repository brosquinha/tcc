# Fake news: emoções e disseminação

O presente trabalho tem como objetivo analisar as reações das pessoas às _fake news_ e relacionar as emoções provocadas por elas com sua propagação.

Autores:
* Lucas Gomes de Paiva
* Thales César Giriboni de Mello e Silva

Orientador:
* Prof. Dr. Ricardo L. de Azevedo Rocha

Código-fonte disponível em: [https://github.com/brosquinha/tcc](https://github.com/brosquinha/tcc)

## Motivação

Dada a importância das notícias falsas nos últimos anos, convém analisar qual o impacto emocional que elas causam nas pessoas. Para tanto, observando o fato de que sua propagação está fortemente relacionada às redes sociais, é possível fazer essa análise utilizando a plataforma Twitter. Hoje, muitas pessoas utilizam a rede social como fonte de notícias para se manterem atualizas.

Além disso, o Twitter possui uma [Interface de Programação de Aplicativo](https://developer.twitter.com/en/docs) (API, do inglês _Application Programming Interface_) madura e fácil de se usar para capturar os dados necessários. Dessa forma, tem-se mais confiança para se capturar dados de reações de pessoas às notícias que elas recebem.

Nos últimos anos, o fenômeno das notícias falsas (comumente conhecidas pelo termo em inglês, _fake news_) tem tido impacto significativo na sociedade. Nas importantes eleições presidenciais dos Estados Unidos da América de 2016, por exemplo, foi estimado que, em média, os adultos daquele país leram e lembraram uma ou mais notícias falsas durante a eleição, com maior exposição a notícias a favor do candidato vencedor, Donald Trump. Como as pessoas são mais inclinadas a acreditarem em histórias que favorecem suas ideologias, e a exposição a notícias falsas tende a aumentar a percepção de veracidade de outras notícias falsas, essas invenções podem prejudicar a habilidade do processo democrático de escolher os candidatos com base nos fatos.

Em 2020, o mundo se viu diante de uma pandemia global, causado pelo coronavírus da mazela Covid-19. Nesse cenário, as diversas notícias falsas que circularam pelo diversos períodos da pandemia se mostraram bastante perigosas, potencialmente colocando a saúde pública em risco.

Também se tem percebido que exposição combinada a notícias falsas podem aumentar nas pessoas as sensações de ineficiência, alienação e cinismo. Com esse cenário, em 2019 foi sancionada uma lei que pune a divulgação de notícias falsas com finalidade eleitoral no Brasil. Isso demonstra quão relevante é a questão das _fake news_ no Brasil e no mundo.

## Metodologia

Por se tratar de um projeto de natureza empírica, escolheu-se uma metodologia de trabalho em que se definiu os passos necessários ao objetivo proposto, mas de forma a possibilitar a experimentação com diferentes _datasets_ e modelos de aprendizado de máquina. 


### Busca de notícias falsas

A primeira parte do projeto consistiu na procura de uma base de notícias falsas diversas que possibilitassem um bom estudo. Com isso, foi escolhido o repositório [_FakeNewsNet_](https://github.com/KaiDMML/FakeNewsNet), que já contém 5877 notícias falsas de fontes variadas, além de mapear _tweets_ que propagam esses notícias.

### Capturas de tweets

O passo seguinte constitui da elaboração de _scripts_ coletores de _tweets_ propagadores e seus _retweets_ e respostas, bem como uma forma de armazená-los para consultas posteriores.

### Treinamento de classificadores

Na terceira etapa do projeto, duas técnicas de _machine learning_ foram estudadas e implementadas para classificar a emoção dos _tweets_ de resposta às notícias falsas: Naive Bayes com utilização de _bag of words_, e um modelo de _deep learning_ em rede neural com LSTM (Long Term Short Memory).

A escolha pelo Naive Bayes se deve à relativa simplicidade do algoritmo, amplamente presente na literatura como exemplo de aprendizado de máquina. Com isso, espera-se obter um desempenho de base mínimo que os demais classificadores treinados com técnicas mais avançadas, como as redes neurais, possam superar. Assim, os classificadores com Naive Bayes servem como parâmetro de comparação para os demais modelos desenvolvidos.

### Validação e testes de confiabilidade

Depois de treinados os modelos de forma satisfatória, é necessário validá-los. Para isso, algumas estratégias foram empregadas de forma a se ter o grau de confiabilidade desses classificadores com bastante segurança.

A primeira estratégia consiste de rodar os classificadores treinados com um _dataset_ em outro _dataset_. Com isso, espera-se que um classificador com alto grau de confiabilidade consiga boas métricas de desempenho em um conjunto de dados que ele nunca viu. O raciocínio por trás dessa estratégia é que um bom classificador apresentará boas métricas de desempenho em cima de dados inéditos, uma vez que essa é a realidade do ambiente em que se pretende executá-lo.

Outra estratégia para validação dos classificadores de emoção é a classificação manual de 1000 respostas ao _tweets_ propagadores. Esse processo é consideravelmente mais trabalhoso e custoso em termos de tempo, mas também dá mais credibilidade para um classificador que conseguir um bom desempenho nesse conjunto de dados.

Durante os testes, observa-se atentamente às matrizes de confusão geradas pelos modelos, procurando obter boas métricas de _recall_ para ambas as classes. Nesse sentido, não interessa um classificador que seja excelente em categorizar uma classe, mas péssimo em outra. Dessa forma, procura-se validar um classificador que seja o melhor possível em ambas as classes.

## Resultados

Abaixo apresentam-se os resultados obtidos pelo projeto. Selecione os modelos desejados utilizando as caixas de seleções na navegação lateral.