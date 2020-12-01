# Fake news: emoções e disseminação

## Instalação

Necessário Python3.6+.

```bash
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Para usar os script de Twitter, prepare um arquivo `.env` com as chaves da API do Twitter:

```bash
cp .env.sample .env
```

Edite os valores de `.env`.

## Scripts de captura de tweets

Para capturar todos os tweets da base FakeNewsNet:

```bash
python3 collect_fakenewsnet_tweets.py >> output.txt 2>&1 &
```

Total de tweets esperados: 763879

## Scripts de classificação de emoções

Para ver as possibilidades de treinamento e testes com os modelos de Naive Bayes:

```bash
python3 run_nb_models.py -h
```

Para ver as possibilidades de treinamento e testes com os modelos de redes neurais:

```bash
python3 run_lstm_models.py -h
```

## Subindo banco de dados PostgreSQL com Docker

Para subir o banco de dados PostgreSQL localmente com docker, rode:

```
docker run --name my_postgres -v /home/user/db_data:/var/lib/postgresql/data -p 5432:5432 -e POSTGRES_PASSWORD=postgres -d postgres
```

Para rodar as migrations:

```
cd datasets/
alembic upgrade head
```

[Tutorial completo de migrations com Alembic](https://alembic.sqlalchemy.org/en/latest/tutorial.html)

### Fazer backup do banco de dados do GCP

* Entrar no Docker do Postgres
* Rodar `pg_dump -U postgres -h IP_SQL_GCP postgres > tweets.sql`
* Para restaurar num banco de dados local chamado tweets: `psql -U postgres tweets < tweets.sql`
* Use `docker cp` para copiar o arquivo tweets.sql para fora do container

## Executando análise das respostas com modelos treinados

Para mais informações sobre os parâmetros necessários para a classificação de emoções das respostas com os modelos já treinados (é necessário que os tweets estejam no banco de dados acessível ao script):

```bash
python3 run_replies_classification.py -h
```
