# -*- coding: utf-8 -*-

'''Sequência de códigos para tratamento de dados inconsistentes e
padronização de dados com escalas diferentes'''

import pandas as pd
base = pd.read_csv('credit_data.csv')

base.describe()  # descreve a base de dados

base.loc[base['age'] < 0]  # busca na base de dados determinada relação

# base.drop(base[base.age < 0].index, implace=True)
# elimina as linhas referentes à instrução

base.mean()  # calcula a média dos valores das colunas

base['age'].mean()
# calcula a média da coluna indicada

base['age'][base.age > 0].mean()
# calcula a média excluindo os valores inconsistentes

base.loc[base.age < 0, 'age'] = 40.92
# substitui os dados inconsistentes pela a média calculada (40.92)

pd.isnull(base['age'])
# mostra quando há valores nulos na coluna

base.loc[pd.isnull(base['age'])]
# encotra os valores nulos de determinada coluna

previsores = base.iloc[:, 1:4].values
# cria a variável "previsores" para as colunas "income, age e loan"

classe = base.iloc[:, 4].values
# cria a variável "classes" para a coluna "c#default"

import numpy as np
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
previsores = imp_mean.fit(previsores).transform(previsores)
# substitui os dados nulos pela a média calculada das respectivas colunas

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
# implementa o modelo de Padronização dos dados para a mesma escala

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)
# cria as variáveis para teste e treinamento com respectivamente 25% e 75% dos dados

'''Sequência de códigos para aprendizagem por Random Forest'''

from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
classificador.fit(previsores_treinamento, classe_treinamento)
# gera a floresta randômica para aprendizado, utilizado 40 árvores para a floresta (n_estimators)

previsoes = classificador.predict(previsores_teste)
''' gera a variável previsões que utiliza o aprendizado para prever, com base
no treinamento, o resultado. Posteriormente é importante comparar esta variável com a 
classe_teste para comparar a quantidade de acertos e erros.'''

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
# compara os valores conhecidos (classe_teste) com os valores previstos pelo algoritmo (previsoes)

matriz = confusion_matrix(classe_teste, previsoes)
# cria a matriz de confusão dos valores conhecidos e previstos

''' previsão de acerto deste algoritmo ficou em 98,40%.'''

