import numpy as np


def monta_conjunto_dados(arquivo: str):
    with open(arquivo, 'r') as arq:
        dados = arq.read()
        dados_organizados = dados.replace(' ', '')

    amostra_aux = np.fromstring(dados_organizados, sep=',')

    return np.reshape(amostra_aux, (1326, 10, 12))
