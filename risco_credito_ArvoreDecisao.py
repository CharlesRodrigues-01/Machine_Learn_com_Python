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

from sklearn.tree import DecisionTreeClassifier, export
classificador = DecisionTreeClassifier(criterion='entropy')
classificador.fit(previsores, classe)
# gera a árvore de decisão para aprendizado

print(classificador.feature_importances_)
# imprime os valores da historia, divida, garantias e renda, calculados através da entropia e do ganho de informação

export._export.export_graphviz(classificador,
                               out_file=('arvore.dot'),
                               feature_names=['história', 'dívida', 'garantias', 'renda'],
                               class_names=['alto', 'moderado', 'baixo'],
                               filled=True,
                               leaves_parallel=True)