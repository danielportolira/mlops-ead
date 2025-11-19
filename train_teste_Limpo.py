#!/usr/bin/env python
# coding: utf-8

# # **Cultura e Práticas em DataOps e MLOps**
# 
# 
# - Por Daniel: realizando teste rodando este código a partir de um ambiente Conda personalizado fornecido pelo professor da cadeira. Hoje é segunda-feira 2025-11-17. Está instalado no meu Notebook Dell Alienware, de nome Antyliah, um banco de dados SQLite para dar suporte à biblioteca MLflow.
# 
# **Autor Professor**: Renan Santos Mendes
# **Email do PROFESSOR**: renansantosmendes@gmail.com
# 
# **Descrição**: Este notebook apresenta um exemplo de uma rede neural profunda com mais de uma camada para um problema de classificação.
# 
# 
# # **Saúde Fetal**
# 
# As Cardiotocografias (CTGs) são opções simples e de baixo custo para avaliar a saúde fetal, permitindo que os profissionais de saúde atuem na prevenção da mortalidade infantil e materna. O próprio equipamento funciona enviando pulsos de ultrassom e lendo sua resposta, lançando luz sobre a frequência cardíaca fetal (FCF), movimentos fetais, contrações uterinas e muito mais.
# 
# Este conjunto de dados contém 2126 registros de características extraídas de exames de Cardiotocografias, que foram então classificados por três obstetras especialistas em 3 classes:
# 
# - Normal
# - Suspeito
# - Patológico

# # 1 - Importando os módulos necessários

import os
import random
import numpy as np
import random as python_random
import tensorflow
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.utils import to_categorical

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import mlflow

# Esta função definida no Python irá inserir seeds propositais de número 42 em todas as funções que utilizam randomização.
# Há desde funções de ambiente operacional até as bibliotecas específicas.
def reset_seeds() -> None:
  """
  Resets the seeds to ensure reproducibility of results.

  This function sets the seed for various random number generation libraries
  to ensure that results are reproducible. The affected libraries are:
  - Python's built-in `random`
  - NumPy
  - TensorFlow

  The seed used is 42.

  Returns:
      None
  """
  os.environ['PYTHONHASHSEED']=str(21)
  tf.random.set_seed(21)
  np.random.seed(21)
  random.seed(21)


# 2 - Fazendo a leitura do dataset e atribuindo às respectivas variáveis

# Aqui, de maneira interessante, o professor referencia a variávei data carregando dados que estão na Web em seu site.
# A biblioteca utilizada para isso é a Pandas com a função read_csv. Data será um objeto pandas.core.frame.DataFrame.
url = 'raw.githubusercontent.com'
username = 'renansantosmendes'
repository = 'lectures-cdas-2023'
file_name = 'fetal_health_reduced.csv'
data = pd.read_csv(f'https://{url}/{username}/{repository}/master/{file_name}')


# 3 - Preparando o dado antes de iniciar o treino do modelo

# Atribuir a X o dataframe data sem a coluna fetal_health. Isto mantém apenas as variáveis preditoras de features em X.
# Em uma única linha fazemos a atribuição a X de uma versão de data com o drop da coluna fetal_health.
# Interessante é que na verdade drop é um método de data!
X=data.drop(["fetal_health"], axis=1)

# Aqui fazemos o complemento: atribuímos a y as variáveis alvo da coluna fetal_health. Para isto, acessamos especificamente
# de data a coluna fetal_health referenciando-a entre colchetes com o nome fetal_health entre aspas.
y=data["fetal_health"]

# Guardar os nomes das colunas para posterior processamento. Interessante mostrar que se retiram as colunas de X, ou seja,
# só haverá as colunas dos features, e não da variável preditora fetal_health.
columns_names = list(X.columns)

# Criando um objeto para pré-processamento dos dados.A classe StandardScaler pertence à scikit-learn.
# Ela padroniza dados, transformando cada feature para que tenha Média = 0 e Desvio Padrão = 1. Esta transformação
# serve para algoritmos de machine learning, como regressão logística, SVM e Redes Neurais.
scaler = preprocessing.StandardScaler()

# Usando o objeto scaler para transformar os valores de X:
X_df = scaler.fit_transform(X)

# Reconstruindo um Datagrame Pandas com os dados padronizados, juntando com os nomes originais das colunas, que deixamos
# guardados em columns_names anteriormente:
X_df = pd.DataFrame(X_df, columns=columns_names)

# Abaixo, clássica instrução de divisão de features e variáveis preditoras em conjuntos de treino e de teste utilizando
# train_test_split, com 30% para teste. A atribuição de 42 ao atributo de entrada random_state permite refazer o teste
# tendo o mesmo resultado de distribuição aleatória do conjunto de dados de teste, permitindo rastrear e repetir o passo caso
# necessário tendo os mesmos resultados de treinamento e teste. X_df é o resultado do processamento feito para permitir
# o treinamento. Recomenda-se comparar os dois objetos, X e X_df.
X_train, X_test, y_train, y_test = train_test_split(X_df,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)

# Isto aqui é curioso: subtrai-se 1 das variáveis preditoras de treinamento e de teste, provavelmente para ajustar rótulos que
# possuem saídas de predição iniciados em 1 para poderem iniciar em zero, mas fica mais claro olhando os resultados depois.
y_train = y_train -1
y_test = y_test - 1


# 4 - Criando o modelo e adicionando camadas

# Criação de camadas de rede neural artificial
# A instrução abaixo chama a função criada antes que opera
# em todas as partes do código e do ambiente que porventura
# façam uso de randomização para que os passos de separações
# ou escolhas aleatórias sejam repetidos. Lembrando a função:
#  os.environ['PYTHONHASHSEED']=str(42): - Define a semente do hash aleatório usado internamente pelo Python
#  tf.random.set_seed(42): Define a semente para o gerador de números aleatórios do TensorFlow.
#  np.random.seed(42): Define a semente para o gerador de números aleatórios do NumPy.
#  random.seed(42): Define a semente para o módulo random da biblioteca padrão do Python.
reset_seeds()

# Aqui define-se uma rede neural sequencial usando a biblioteca Keras.
# Para constar, a importação foi feita assim:
# from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Dense, InputLayer
# from keras.utils import to_categorical

# Criando um modelo Sequential, que é uma pilha linear de camadas.
# Cada camada tem exatamente uma entrada e uma saída???????
model = Sequential()

# Definindo a forma de entrada da rede;
# input_shape = (X_train.shape[1],) significa que a entrada será um vetor com
# o mesmo número de colunas (features) que o conjunto de treino X_train.
# Esta camada não possui neurônios de ativação. A função dele é
# estabelecer para a rede neural o formado de entrada dos dados.
#model.add(InputLayer(input_shape=(X_train.shape[1], )))
model.add(InputLayer(shape=(X_train.shape[1], )))

# Adiciona a primeira camada oculta no modo Dense que é totalmente conectada.
# Especifica que esta camada possui 10 neurônios e que usará a função de
# ativação ReLU - Rectified Linear Unit, que introduz não linearidade e ajuda
# a rede a aprender padrões complexos.
model.add(Dense(units=10, activation='relu'))

# Adiciona mais uma camada oculta igual à anterior:
model.add(Dense(units=10, activation='relu'))

# Finalmente, adicionando uma camada final de 3 neurônios que representa os 3
# possíveis estados da variável alvo fetal_health. A função de ativação softmax
# transforma as saídas em probabilidades que somam 1, o que é ideal para a
# classificação multiclasse.
model.add(Dense(units=3, activation='softmax'))


# 5 - Compilando o modelo

# Configurando o modelo de rede neural para treinamento utilizando o método compile():

# - loss = 'sparse_categorical_cossentropy': define a função de perda (loss function) a qual mede o erro entre as previsões
# e os rótulos reais. a função de perda escolhida, sparse_categorical_crossentropy é usada para classificação multiclasse
# quando os rótulos são inteiros, tais como 0, 1, 2 e não vetores # one-hot.

# - optimizer = 'adam': Define o otimizador, que ajusta os pesos da rede para minimizar a função de perda. Adam - Adaptive
# Moment Estimation é um dos otimizadores mais populares.

# - metrics = ['accuraccy']: define a métrica de avaliação usada durante o treinamento e validação. accuracy = (número de
# acertos)/(total de exemplos).

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# Configurando o MLflow

# Aqui tinha muita coisa pessoal do Professor Renan. Depois adaptei para rodar localmente.
# Para isso, antes de rodar o código, lembrar de digitar 'mlflow ui' em um prompt de comando para ativar a home page
# que vai apresentar os resultados da biblioteca MLflow.

#os.environ['MLFLOW_TRACKING_USERNAME'] = 'renansantosmendes'
#os.environ['MLFLOW_TRACKING_PASSWORD'] = '6d730ef4a90b1caf28fbb01e5748f0874fda6077'
#mlflow.set_tracking_uri('https://dagshub.com/renansantosmendes/puc_lectures_mlops.mlflow')
mlflow.set_tracking_uri("http://localhost:5000") # Vai usar o banco de dados SWLite e o MLflow no meu própro computador.

mlflow.keras.autolog(log_models=True,
                     log_input_examples=True,
                     log_model_signatures=True)


# 6 - Executando o treino do modelo

# Iniciando experimento MLflow com o nome 'experiment_mlops_ead.
# Tudo o que acontece dentro do bloco 'with' será registrado automaticamente
# pelo MLflow. No caso, serão parâmetros do modelo, métricas de desempenho
# e os artefatos, que no caso é o modelo treinado. O objeto 'run' representa
# a execução atual, permitindo aceso a informações como ID, status e
# resultados.

with mlflow.start_run(run_name='experiment_mlops_ead') as run:
  model.fit(X_train,
            y_train,
            epochs=50,
            validation_split=0.2,
            verbose=3)
