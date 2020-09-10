# -*- coding: utf-8 -*-

'''Sequência de códigos para aprendizagem através de Regressão Logística'''

import pandas as pd
base = pd.read_csv('risco_credito2.csv')

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

from sklearn.linear_model import LogisticRegression
classificador = LogisticRegression()
classificador.fit(previsores, classe)
# gera o algoritmo para aprendizado

print(classificador.intercept_)
print(classificador.coef_)

resultado = classificador.predict([[0,0,1,2], [2, 0, 0, 0]])
resultado2 = classificador.predict_proba([[0,0,1,2], [2, 0, 0, 0]])

print(resultado)
print(resultado2)