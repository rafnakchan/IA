import numpy as np


def monta_conjunto_dados(arquivo: str) -> np.ndarray:
    with open(arquivo, 'r') as arq:
        dados = arq.read()
        dados_organizados = dados.replace(' ', '')

    amostra_aux = np.fromstring(dados_organizados, sep=',')

    return np.reshape(amostra_aux, (1326, 10, 12))


def monta_rotulo(arquivo: str) -> np.ndarray:
    entrada_rotulo = np.loadtxt(arquivo, dtype=str)

    mapeamento_rotulo = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
                         'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
                         'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}

    rotulo = [[0.0 for j in range(26)] for i in range(1326)]
    for i in range(0, 1326):
        for j in range(0, 26):
            if mapeamento_rotulo[entrada_rotulo[j]] == (i % 26):
                rotulo[i][j] = 1.0
            else:
                rotulo[i][j] = 0.0

    return np.asarray(rotulo)


def verifica_resultado(resultado_obtido: np.ndarray, resultado_esperado: np.ndarray) -> (bool, str, str):
    mapeamento_rotulo = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                         10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
                         19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

    letra_esperada: str = ''
    letra_predita: str = ''
    acerto: bool = False

    for i in range(0, 26):
        if resultado_esperado[i] == 1:
            letra_esperada = mapeamento_rotulo[i]
        if resultado_obtido[i] > 0.9:
            if letra_predita == '':
                letra_predita = mapeamento_rotulo[i]
            else:
                letra_predita = 'Mais de uma letra predita'

    if letra_esperada == letra_predita:
        acerto = True

    return acerto, letra_esperada, letra_predita
