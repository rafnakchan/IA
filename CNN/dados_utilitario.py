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

    # Modelagem dos dados de entrada
    (conjunto_treino, rotulo_treino), (conjunto_aux, rotulo_aux) = mnist.load_data()
    rotulo_treino = to_categorical(rotulo_treino)
    rotulo_aux = to_categorical(rotulo_aux)

    conjunto_teste, conjunto_validacao = train_test_split(conjunto_aux, test_size=0.1, random_state=42, shuffle=False)
    rotulo_teste, rotulo_validacao = train_test_split(rotulo_aux, test_size=0.1, random_state=42, shuffle=False)

    return (conjunto_treino, rotulo_treino), (conjunto_teste, rotulo_teste), (conjunto_validacao, rotulo_validacao)


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

    return (conjunto_treino, rotulo_treino), (conjunto_teste, rotulo_teste), (conjunto_validacao, rotulo_validacao)


def grava_arquivo_pesos(pesos: np.ndarray, nome_arquivo: str):
    # montagem do conjunto de dados
    with open('./Arquivos/Resultados/' + nome_arquivo + '.txt', 'w') as arq:
        for registro in pesos:
            arq.write(np.array2string(registro) + '\n')
    return


def grava_arquivo_resultado(acerto: list[bool], valor_predito: list[str], valor_esperado: list[str], total_acertos, percentual_acertos, matriz: tuple, nome_arquivo, linhas):
    # montagem do conjunto de dados
    with open('./Arquivos/Resultados/' + nome_arquivo + '.md', 'w') as arq:
        arq.write('# Resultados do Treinamento\n\n')
        arq.write('## Total de Acertos:\n')
        arq.write('- ' + str(total_acertos) + '\n\n')
        arq.write('## Percentual de Acertos:\n')
        arq.write('- ' + str(percentual_acertos) + ' %\n\n')
        arq.write('## Matriz de Confusao:\n')
        arq.write('|        | Predito 1 | Predito 0 |\n')
        arq.write('|--------|-----------|-----------|\n')
        arq.write('| Real 1 | ' + str(matriz[0]).ljust(9) + ' | ' + str(matriz[1]).ljust(9) + ' |\n')
        arq.write('| Real 0 | ' + str(matriz[2]).ljust(9) + ' | ' + str(matriz[3]).ljust(9) + ' |\n\n')
        arq.write('## Tabela de Acertos:\n')
        arq.write('| Valor Predito | Valor Esperado | Acerto |\n')
        arq.write('|---------------|----------------|--------|\n')
        for i in range(0, linhas):
            arq.write('| ' + valor_predito[i].ljust(13) + ' | ' + valor_esperado[i].ljust(14) + ' | ' + str(acerto[i]).ljust(6) + ' |\n')
    return
