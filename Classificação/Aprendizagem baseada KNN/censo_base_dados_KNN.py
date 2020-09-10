# -*- coding: utf-8 -*-

'''Sequência de códigos para conversão de variáveis e padronização de
dados com escalas diferentes'''

import pandas as pd
base = pd.read_csv('census.csv')

previsores = base.iloc[:, 0:14].values
# cria a variável "previsores" para as colunas indicadas

classe = base.iloc[:, 14].values
# cria a variável "classes" para a coluna indicada

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsores = LabelEncoder()
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])
'''transforma as variáveis categóricas(nominais e ordinais) para variáveis
numéricas(discretas)'''

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), 
    [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
previsores = ct.fit_transform(previsores).toarray()
'''cria arrays para transformar  as variáveis numéricas-discretas para arrays 
de valores 0 e 1'''

labelEncoder_classe = LabelEncoder()
classe = labelEncoder_classe.fit_transform(classe)
'''transforma as variáveis categóricas para variáveis
numéricas(discretas)'''

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
# faz o escalonamento padronizado dos dados

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)
# cria as variáveis para teste e treinamento com respectivamente 15% e 85% dos dados

'''Sequência de códigos para aprendizagem por Instâncias (KNN)'''

from sklearn.neighbors import KNeighborsClassifier
classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classificador.fit(previsores_treinamento, classe_treinamento)
# gera a o algoritmo KNN para aprendizado

previsoes = classificador.predict(previsores_teste)
''' gera a variável previsões que utiliza o aprendizado para prever, com base
no treinamento, o resultado. Posteriormente é importante comparar esta variável com a 
classe_teste para comparar a quantidade de acertos e erros.'''

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
# compara os valores conhecidos (classe_teste) com os valores previstos pelo algoritmo (previsoes)

matriz = confusion_matrix(classe_teste, previsoes)
# cria a matriz de confusão dos valores conhecidos e previstos

''' previsão de acerto deste algoritmo ficou em 82,23%.'''