# -*- coding: utf-8 -*-


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
# calcula a média excluindo os valores errados

base.loc[base.age < 0, 'age'] = 40.92
# substitui os dados incorretos pela a média calculada (40.92)

