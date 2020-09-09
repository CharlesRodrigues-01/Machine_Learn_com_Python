# -*- coding: utf-8 -*-

'''Sequência de códigos para aprendizagem bayesiana (Naives Bayes)'''

import pandas as pd
base = pd.read_csv('risco_credito.csv')

previsores = base.iloc[:, 0:4].values
# cria a variável previsores
classe = base.iloc[:, 4].values
# cria a variável classe

from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
previsores[:, 0] = labelEncoder.fit_transform(previsores[:, 0])
previsores[:, 1] = labelEncoder.fit_transform(previsores[:, 1])
previsores[:, 2] = labelEncoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelEncoder.fit_transform(previsores[:, 3])
'''transforma as variáveis categóricas para variáveis
numéricas(discretas)'''

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores, classe)
# gera a tabela de probabilidade para aprendizado

resultado = classificador.predict([[0,0,1,2], [2, 0, 0, 0]])
''''mostra o resultado através da inserção dos indicadores, baseado no aprendizado
valores atribuidos: hist-boa, div-alta, garant-nenhuma, renda-acima_35'''

print(classificador.classes_)
# imprime as categorias das classes
print(classificador.class_count_)
# imprime a quantidade total de cada classe
print(classificador.class_prior_)
# imprime a probabilidade de cada classe