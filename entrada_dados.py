import numpy as np


def monta_conjunto_dados(arquivo: str):
    with open(arquivo, 'r') as arq:
        dados = arq.read()
        dadosOrganizados = dados.replace(' ', '').replace('-1', '0')

    rotulo = [None] * 1326
    for i in range(0, 1326):
        rotulo[i] = np.remainder(i, 26)

    rotulo = np.reshape(rotulo, (1326, 1))
    amostraAux = np.fromstring(dadosOrganizados, sep=',')

    return np.hstack((np.reshape(amostraAux, (1326, 120)), rotulo))
