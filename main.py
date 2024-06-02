import matplotlib.pyplot as plt
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

conjunto_teste, conjunto_validacao = train_test_split(resto, test_size=0.5, random_state=42, shuffle=False)

# definicao da base da arquitetura da mlp
####################################################################

# passo 1 - inicializacao
numero_epoca = 0
numero_maximo_epocas = 500
taxaDeAprendizado = 0.5
numeroNeuroniosEscondidos = 50
erro_quadrado_validacao = [0.0 for _ in range(numero_maximo_epocas)]
percentual_erros_validacao = [0.0 for _ in range(numero_maximo_epocas)]
erro_quadrado_medio = [0.0 for _ in range(numero_maximo_epocas)]
variacao_erro_quadrado = [0.0 for _ in range(numero_maximo_epocas)]

# neuronios da camada escondida
camada_escondida = [Neuronio(120) for _ in range(numeroNeuroniosEscondidos)]

# neuronios da camada de saida
camada_saida = [Neuronio(numeroNeuroniosEscondidos) for _ in range(26)]

# passo 2 - executa mais uma epoca
while True:

    # passagem pelo conjunto de treinamento
    numero_amostra = 0
    erro_total_epoca = 0.0
    for amostra_rotulada in conjunto_treino:
        # passo 3 - preparando entrada
        # amostra_rotulada é um array de tamanho 121,
        # nos índices 0 a 119 contem as 120 entradas e no indice 120 o rótulo
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
            # passando por todos o resultados da camada de saída,
            # se a saida for do mesmo índice do rótulo, a saida esperada é 1
            if (i == rotulo):
                erros_locais[i] = 1 - saidas_finais[i]
            else:
                erros_locais[i] = 0 - saidas_finais[i]

            # valor de entrada do neuronio
            entrada_neuronio = camada_saida[i].somatoria_entradas(saidas_camada_escondida)
            # calculo do "deltinha"
            gradiente_local[i] = erros_locais[i] * funcao_ativacao.derivada_sigmoide(entrada_neuronio)
            # calculo da correcao de pesos das entradas da camada de saida
            for j in range(0, numeroNeuroniosEscondidos):
                correcao_pesos_camada_saida[i][j] = taxaDeAprendizado * gradiente_local[i] * saidas_camada_escondida[j]
            # correcao do bias
            correcao_pesos_camada_saida[i][numeroNeuroniosEscondidos] = taxaDeAprendizado * gradiente_local[i]
            # somando o erro total da amostra para calculo do erro quadrado medio no final da epoca
            erro_total_amostra = erro_total_amostra + (np.power(erros_locais[i], 2)) / 2

        # passo 7 - retro propagação do erro
        correcao_pesos_camada_escondida = [[0 for x in range(121)] for y in range(numeroNeuroniosEscondidos)]
        for i in range(0, numeroNeuroniosEscondidos):
            delta = 0.0
            for j in range(0, 26):
                # lista de pesos da camada de saida
                pesos = camada_saida[j].pesos
                # multiplica o "deltinha" da camada de saida pelo peso correspondente a saida do neurônio
                # soma tudo em um novo "deltinha" para o neuronio da camada escondida
                delta = delta + (gradiente_local[j] * pesos[i])
            # calculo da correcao de pesos das entradas da camada escondida
            for j in range(0, 120):
                correcao_pesos_camada_escondida[i][j] = taxaDeAprendizado * delta * amostra[j]
            # correcao do bias
            correcao_pesos_camada_escondida[i][120] = taxaDeAprendizado * delta

        erro_total_epoca += erro_total_amostra

        # passo 8 - atualizacao pesos
        for i in range(0, 26):
            camada_saida[i].atualizar_pesos(correcao_pesos_camada_saida[i])
        for i in range(0, numeroNeuroniosEscondidos):
            camada_escondida[i].atualizar_pesos(correcao_pesos_camada_escondida[i])

        numero_amostra += 1

    # passagem pelo conjunto de validacao
    total_erros_validacao = 0
    for amostra_rotulada_validacao in conjunto_validacao:
        amostra_validacao = np.delete(amostra_rotulada_validacao, 120, 0)
        rotulo_validacao = amostra_rotulada_validacao[120]
        # feedfoward
        saidas_camada_escondida_validacao = [0 for _ in range(numeroNeuroniosEscondidos)]
        for i in range(0, numeroNeuroniosEscondidos):
            saidas_camada_escondida_validacao[i] = camada_escondida[i].ativacao(amostra_validacao)

        saidas_finais_validacao = [0 for _ in range(26)]
        for i in range(0, 26):
            saidas_finais_validacao[i] = camada_saida[i].ativacao(saidas_camada_escondida_validacao)
        # checagem de erro
        erro = False
        for i in range(0, 26):
            if i == rotulo_validacao:
                erros_local_validacao = 1 - saidas_finais_validacao[i]
                if saidas_finais_validacao[i] < 0.9:
                    erro = True
            else:
                erros_local_validacao = 0 - saidas_finais_validacao[i]
                if saidas_finais_validacao[i] > 0.1:
                    erro = True
            erro_quadrado_validacao[numero_epoca] += ((np.power(erros_local_validacao, 2)) / 2)

        if erro:
            total_erros_validacao += 1

    erro_quadrado_medio[numero_epoca] = erro_total_epoca / numero_amostra
    percentual_erros_validacao[numero_epoca] = total_erros_validacao / 130
    try:
        variacao_erro_quadrado = np.abs(erro_quadrado_medio[numero_epoca] - erro_quadrado_medio[numero_epoca - 1])
    except:
        variacao_erro_quadrado = erro_quadrado_medio[numero_epoca]

    print('********************************************************************')
    print('numero epoca: ' + str(numero_epoca + 1))
    print('erro quadrado medio: ' + str(erro_quadrado_medio[numero_epoca]))
    print('variacao erro quadrado: ' + str(variacao_erro_quadrado))
    print('erro quadrado na validacao: ' + str(erro_quadrado_validacao[numero_epoca]))
    print('percentual de erros na validacao: ' + str(percentual_erros_validacao[numero_epoca] * 100))

    # passo 9 - condicao de parada
    baixo_erro_validacao = percentual_erros_validacao[numero_epoca] < 0.2
    baixo_erro_quadrado = erro_quadrado_medio[numero_epoca] < 0.01
    numero_epoca += 1
    limite_epocas = numero_epoca >= numero_maximo_epocas
    if limite_epocas | baixo_erro_validacao | baixo_erro_quadrado:
        break

    np.random.shuffle(conjunto_treino)

# passagem pelo conjunto de teste
total_erros_teste = 0
verdadeiro_positivo = 0
falso_positivo = 0
verdadeiro_negativo = 0
falso_negativo = 0
for amostra_rotulada_teste in conjunto_teste:
    amostra_teste = np.delete(amostra_rotulada_teste, 120, 0)
    rotulo_teste = amostra_rotulada_teste[120]

    saidas_camada_escondida_teste = [0 for _ in range(numeroNeuroniosEscondidos)]
    for i in range(0, numeroNeuroniosEscondidos):
        saidas_camada_escondida_teste[i] = camada_escondida[i].ativacao(amostra_teste)

    saidas_finais_teste = [0 for _ in range(26)]
    for i in range(0, 26):
        saidas_finais_teste[i] = camada_saida[i].ativacao(saidas_camada_escondida_teste)

    erro_teste = False
    for i in range(0, 26):
        if i == rotulo_teste:
            if saidas_finais_teste[i] < 0.9:
                falso_negativo += 1
                erro_teste = True
            else:
                verdadeiro_positivo += 1
        else:
            if saidas_finais_teste[i] > 0.1:
                falso_positivo += 1
                erro_teste = True
            else:
                verdadeiro_negativo += 1
    if erro_teste:
        total_erros_teste += 1

percentual_erros_teste = total_erros_teste / 130
print('********************************************************************')
print('erros na Teste: ' + str(total_erros_teste))
print('percentual de erros na Teste: ' + str(percentual_erros_teste * 100))
print('verdadeiro_positivo: ' + str(verdadeiro_positivo))
print('falso_positivo: ' + str(falso_positivo))
print('verdadeiro_negativo: ' + str(verdadeiro_negativo))
print('falso_negativo: ' + str(falso_negativo))

plt.plot(erro_quadrado_medio, marker='o', color='r', linewidth='1.0')
plt.plot(percentual_erros_validacao, marker='o', color='b', linewidth='1.0')
plt.xlabel("Epoca")
plt.grid()
plt.show()

plt.plot(erro_quadrado_medio, marker='o', color='r', linewidth='1.0')
plt.xlabel("Epoca")
plt.ylabel("Erro Quadrado Medio")
plt.grid()
plt.show()

plt.plot(erro_quadrado_validacao, marker='o', color='g', linewidth='1.0')
plt.xlabel("Epoca")
plt.ylabel("Erro Quadrado Validacao")
plt.grid()
plt.show()

plt.plot(percentual_erros_validacao, marker='o', color='b', linewidth='1.0')
plt.xlabel("Epoca")
plt.ylabel("Percentual Erros Validacao")
plt.grid()
plt.show()
