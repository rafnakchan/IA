import numpy as np
from sklearn.model_selection import train_test_split

import entrada_dados
import funcao_ativacao
from neuronio import Neuronio

# importacao dos dados
####################################################################

# configuração para imprimir todo o array
np.set_printoptions(threshold=np.inf)

# remodelando os dados que ficaram em 1 dimensao para uma matriz
conjunto = entrada_dados.monta_conjunto_dados('X.txt')

conjunto_treino, resto = train_test_split(conjunto, train_size=1066, random_state=42, shuffle=False)

conjunto_validacao, conjunto_teste = train_test_split(resto, test_size=0.5, random_state=42, shuffle=False)

# definicao da base da arquitetura da mlp
####################################################################

# passo 1 - inicializacao
numeroEpoca = 0
limiteEpoca = 50
taxaDeAprendizado = 0.01
numeroNeuroniosEscondidos = 13

# neuronios da camada escondida
camada_escondida = [Neuronio(120) for _ in range(numeroNeuroniosEscondidos)]

# neuronios da camada de saida
camada_saida = [Neuronio(numeroNeuroniosEscondidos) for _ in range(26)]

while True:
    # passos 2 e 9 - condição de parada
    if (numeroEpoca >= limiteEpoca):
        break

    numero_amostra = 0
    erro_total_epoca = 0.0
    for amostra_rotulada in conjunto_treino:
        # passo 3 - preparando entrada
        amostra = np.delete(amostra_rotulada, 120, 0)
        rotulo = amostra_rotulada[120]

        # passo 4 - camada escondida
        saidas_camada_escondida = [0 for _ in range(numeroNeuroniosEscondidos)]
        for i in range(0, numeroNeuroniosEscondidos):
            saidas_camada_escondida[i] = camada_escondida[i].ativacao(amostra)

        # passo 5 camada de saida
        saidas_finais = [0 for _ in range(26)]
        for i in range(0, 26):
            saidas_finais[i] = camada_saida[i].ativacao(saidas_camada_escondida)

        # passo 6 calcula erros
        erros_locais = [0 for _ in range(26)]
        erro_total_amostra = 0.0
        gradiente_local = [0 for _ in range(26)]
        correcao_pesos_camada_saida = [[0 for x in range(numeroNeuroniosEscondidos + 1)] for y in range(26)]
        for i in range(0, 26):
            if (i == rotulo):
                erros_locais[i] = 1 - saidas_finais[i]
            else:
                erros_locais[i] = 0 - saidas_finais[i]

            entrada_neuronio = camada_saida[i].somatoria_entradas(saidas_camada_escondida)
            gradiente_local[i] = erros_locais[i] * funcao_ativacao.derivada_sigmoide(entrada_neuronio)
            for j in range(0, numeroNeuroniosEscondidos):
                correcao_pesos_camada_saida[i][j] = taxaDeAprendizado * gradiente_local[i] * saidas_camada_escondida[j]
            correcao_pesos_camada_saida[i][numeroNeuroniosEscondidos] = taxaDeAprendizado * gradiente_local[i]

            erro_total_amostra = erro_total_amostra + (np.power(erros_locais[i], 2)) / 2

        # passo 7 - retro propagação
        correcao_pesos_camada_escondida = [[0 for x in range(121)] for y in range(numeroNeuroniosEscondidos)]
        for i in range(0, numeroNeuroniosEscondidos):
            delta = 0.0
            for j in range(0, 26):
                pesos = camada_saida[j].pesos
                delta = delta + (gradiente_local[j] * pesos[i])
            for j in range(0, 120):
                correcao_pesos_camada_escondida[i][j] = taxaDeAprendizado * delta * amostra[j]
            correcao_pesos_camada_escondida[i][120] = taxaDeAprendizado * delta

        # passo 8 - atualizacao pesos
        for i in range(0, 26):
            camada_saida[i].atualizar_pesos(correcao_pesos_camada_saida[i])
        for i in range(0, numeroNeuroniosEscondidos):
            camada_escondida[i].atualizar_pesos(correcao_pesos_camada_escondida[i])

        erro_total_epoca += erro_total_amostra
        numero_amostra += 1
        # print('valor instantaneo erro: ' + str(erro_total_amostra))

    erro_quadrado_medio = erro_total_epoca / numero_amostra
    numeroEpoca += 1
    print('numero epoca: ' + str(numeroEpoca))
    print('erro quadrado medio: ' + str(erro_quadrado_medio))
    np.random.shuffle(conjunto_treino)

    total_erros = 0
    for amostra_rotulada in conjunto_teste:
        amostra = np.delete(amostra_rotulada, 120, 0)
        rotulo = amostra_rotulada[120]

        saidas_camada_escondida = [0 for _ in range(numeroNeuroniosEscondidos)]
        for i in range(0, numeroNeuroniosEscondidos):
            saidas_camada_escondida[i] = camada_escondida[i].ativacao(amostra)

        saidas_finais = [0 for _ in range(26)]
        for i in range(0, 26):
            saidas_finais[i] = camada_saida[i].ativacao(saidas_camada_escondida)

        erro = False
        for i in range(0, 26):
            if (i == rotulo):
                if (saidas_finais[i] < 0.7):
                    erro = True
            else:
                if (saidas_finais[i] > 0.3):
                    erro = True
        if (erro):
            total_erros += 1

    percentual_erros = total_erros / 130
    print("percentual de erros: " + str(percentual_erros))
