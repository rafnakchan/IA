import numpy as np
from sklearn.model_selection import train_test_split
from keras.api.datasets import mnist
from keras.api.utils import to_categorical


def monta_conjunto_dados(dados_mnist: bool):
    if dados_mnist:
        return monta_conjunto_dados_mnist()
    else:
        return monta_conjunto_dados_ep1()


def monta_conjunto_dados_mnist():
    (conjunto_treino, rotulo_treino), (conjunto_aux, rotulo_aux) = mnist.load_data()
    rotulo_treino = to_categorical(rotulo_treino)
    rotulo_aux = to_categorical(rotulo_aux)

    conjunto_teste, conjunto_validacao = train_test_split(conjunto_aux, test_size=0.1, random_state=42, shuffle=False)
    rotulo_teste, rotulo_validacao = train_test_split(rotulo_aux, test_size=0.1, random_state=42, shuffle=False)

    return conjunto_treino, rotulo_treino, conjunto_teste, rotulo_teste, conjunto_validacao, rotulo_validacao


def monta_conjunto_dados_ep1():
    # montagem do conjunto de dados
    with open('./Arquivos/X.txt', 'r') as arq:
        dados = arq.read()
        dados_organizados = dados.replace(' ', '')

    conjunto_aux = np.fromstring(dados_organizados, sep=',')
    conjunto_treino, conjunto_aux = train_test_split(np.reshape(conjunto_aux, (1326, 10, 12)), train_size=1066,
                                                     random_state=42, shuffle=False)
    conjunto_teste, conjunto_validacao = train_test_split(conjunto_aux, test_size=0.5, random_state=42, shuffle=False)

    # montagem do rotulo
    entrada_rotulo = np.loadtxt('./Arquivos/Y_letra.txt', dtype=str)
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

    rotulo_treino, rotulo_aux = train_test_split(np.asarray(rotulo), train_size=1066, random_state=42, shuffle=False)
    rotulo_teste, rotulo_validacao = train_test_split(rotulo_aux, test_size=0.5, random_state=42, shuffle=False)

    return conjunto_treino, rotulo_treino, conjunto_teste, rotulo_teste, conjunto_validacao, rotulo_validacao


def verifica_resultado(resultado_obtido: np.ndarray, resultado_esperado: np.ndarray, dados_mnist: bool) -> (bool, str, str):
    if dados_mnist:
        return verifica_resultado_mnist(resultado_obtido, resultado_esperado)
    else:
        return verifica_resultado_ep1(resultado_obtido, resultado_esperado)


def verifica_resultado_ep1(resultado_obtido: np.ndarray, resultado_esperado: np.ndarray) -> (bool, str, str):
    mapeamento_rotulo = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                         10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
                         19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

    letra_esperada: str = ''
    letra_predita: str = ''
    acerto: bool = False

    for i in range(0, 26):
        if resultado_esperado[i] == 1:
            letra_esperada = mapeamento_rotulo[i]
        if resultado_obtido[i] >= 0.9:
            if letra_predita == '':
                letra_predita = mapeamento_rotulo[i]
            else:
                letra_predita = 'Mais de uma letra predita'

    if letra_esperada == letra_predita:
        acerto = True

    return acerto, letra_esperada, letra_predita


def verifica_resultado_mnist(resultado_obtido: np.ndarray, resultado_esperado: np.ndarray) -> (bool, str, str):
    numero_esperado: str = ''
    numero_predito: str = ''
    acerto: bool = False

    for i in range(0, 10):
        if resultado_esperado[i] == 1:
            numero_esperado = str(i)
        if resultado_obtido[i] >= 0.9:
            if numero_predito == '':
                numero_predito = str(i)
            else:
                numero_predito = 'Mais de um numero predito'

    if numero_esperado == numero_predito:
        acerto = True

    return acerto, numero_esperado, numero_predito


def grava_arquivo_pesos(pesos: np.ndarray, nome_arquivo: str):
    # montagem do conjunto de dados
    with open('./Arquivos/Resultados/' + nome_arquivo + '.txt', 'w') as arq:
        for registro in pesos:
            arq.write(np.array2string(registro) + '\n')
    return


def grava_arquivo_resultado(acerto: list[bool], valor_predito: list[str], valor_esperado: list[str], nome_arquivo: str, linhas: int):
    # montagem do conjunto de dados
    with open('./Arquivos/Resultados/' + nome_arquivo + '.md', 'w') as arq:
        arq.write('| Valor Predito | Valor Esperado | Acerto |\n')
        arq.write('|---------------|----------------|--------|\n')
        for i in range(0, linhas):
            arq.write('| ' + valor_predito[i].ljust(13) + ' | ' + valor_esperado[i].ljust(14) + ' | ' + str(acerto[i]).ljust(6) + ' |\n')
    return
