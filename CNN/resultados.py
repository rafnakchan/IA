import numpy as np


def verifica_resultado(resultado_obtido: np.ndarray, resultado_esperado: np.ndarray, dados_mnist: bool):
    if dados_mnist:
        return verifica_resultado_mnist(resultado_obtido, resultado_esperado)
    else:
        return verifica_resultado_ep1(resultado_obtido, resultado_esperado)


def verifica_resultado_ep1(resultado_obtido: np.ndarray, resultado_esperado: np.ndarray):
    mapeamento_rotulo = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                         10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
                         19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

    letra_esperada: str = ''
    letra_predita: str = ''
    acerto: bool = False

    verdadeiro_positivo: int = 0
    falso_positivo: int = 0
    falso_negativo: int = 0
    verdadeiro_negativo: int = 0

    for i in range(0, 26):
        if resultado_esperado[i] == 1:
            letra_esperada = mapeamento_rotulo[i]
            if resultado_obtido[i] >= 0.9:
                verdadeiro_positivo += 1
            else:
                falso_positivo += 1
        else:
            if resultado_obtido[i] <= 0.1:
                verdadeiro_negativo += 1
            else:
                falso_negativo += 1
        if resultado_obtido[i] >= 0.9:
            if letra_predita == '':
                letra_predita = mapeamento_rotulo[i]
            else:
                letra_predita = 'Mais de uma letra predita'

    if letra_esperada == letra_predita:
        acerto = True

    matriz = (verdadeiro_positivo, falso_positivo, falso_negativo, verdadeiro_negativo)
    return acerto, (letra_esperada, letra_predita), matriz


def verifica_resultado_mnist(resultado_obtido: np.ndarray, resultado_esperado: np.ndarray):
    numero_esperado: str = ''
    numero_predito: str = ''
    acerto: bool = False

    verdadeiro_positivo: int = 0
    falso_positivo: int = 0
    falso_negativo: int = 0
    verdadeiro_negativo: int = 0

    for i in range(0, 10):
        if resultado_esperado[i] == 1:
            numero_esperado = str(i)
            if resultado_obtido[i] >= 0.9:
                verdadeiro_positivo += 1
            else:
                falso_positivo += 1
        else:
            if resultado_obtido[i] <= 0.1:
                verdadeiro_negativo += 1
            else:
                falso_negativo += 1
        if resultado_obtido[i] >= 0.9:
            if numero_predito == '':
                numero_predito = str(i)
            else:
                numero_predito = 'Mais de um numero predito'

    if numero_esperado == numero_predito:
        acerto = True

    matriz_amostra = (verdadeiro_positivo, falso_positivo, falso_negativo, verdadeiro_negativo)
    return acerto, (numero_esperado, numero_predito), matriz_amostra
