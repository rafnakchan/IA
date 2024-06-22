import numpy as np


def monta_conjunto_dados(arquivo: str):
    with open(arquivo, 'r') as arq:
        dados = arq.read()
        dados_organizados = dados.replace(' ', '')

    amostra_aux = np.fromstring(dados_organizados, sep=',')

    return np.reshape(amostra_aux, (1326, 10, 12))


def monta_rotulo(arquivo: str):
    entrada_rotulo = np.loadtxt(arquivo, dtype=str)

    mapeamento_rotulo = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
                        'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
                        'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}

    rotulo = [[0 for j in range(26)] for i in range(1326)]
    for i in range(0, 1326):
        for j in range(0, 26):
            if mapeamento_rotulo[entrada_rotulo[j]] == (i % 26):
                rotulo[i][j] = 1
            else:
                rotulo[i][j] = 0

    return rotulo
