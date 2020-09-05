# -*- coding: utf-8 -*-

'''Sequência de códigos para conversão de variáveis e padronização de
dados com escalas diferentes'''

import pandas as pd
base = pd.read_csv('census.csv')

previsores = base.iloc[:, 0:14].values
# cria a variável "previsores" para as colunas indicadas

classe = base.iloc[:, 14].values
# cria a variável "classes" para a coluna indicada

from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])
'''transforma as variáveis categóricas(nominais) para variáveis
numéricas(discretas)'''
