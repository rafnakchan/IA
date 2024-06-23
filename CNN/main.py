import time

import numpy as np
from keras.api.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.api.models import Sequential

import dados_utilitario
import resultados

# importacao dos dados
####################################################################

# Mudar para False se desejar usar os dados de caracteres do EP 1 de MLP
dados_mnist: bool = True
(conjunto_treino, rotulo_treino), (conjunto_teste, rotulo_teste), (conjunto_validacao, rotulo_validacao) = (
    dados_utilitario.monta_conjunto_dados(dados_mnist))

# configuracoes
time = time.time()
np.set_printoptions(threshold=np.inf)
epocas: int = 5
tamanho_amostra_validacao: int = np.size(conjunto_validacao, axis=0)
filtros_primeira_camada: int = 64
filtros_segunda_camada: int = 32
tamanho_kernel: int = 3
tamanho_pooling: int = 2
formato_imagem: tuple[int, int, int] = (np.size(conjunto_treino, axis=1), np.size(conjunto_treino, axis=2), 1)
neuronios_camada_saida: int = np.size(rotulo_treino, axis=1)

# montagem do modelo
model = Sequential()

# Camadas de Kernel e Pooling
model.add(Conv2D(filtros_primeira_camada, kernel_size=tamanho_kernel, activation='relu', input_shape=formato_imagem))
if dados_mnist:
    model.add(MaxPool2D(pool_size=tamanho_pooling))
model.add(Conv2D(filtros_segunda_camada, kernel_size=tamanho_kernel, activation='relu'))
if dados_mnist:
    model.add(MaxPool2D(pool_size=tamanho_pooling))

# Camada de Flatten
model.add(Flatten())
# camada Densa de saída
model.add(Dense(neuronios_camada_saida, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# treinamento
pesos_inicial = model.get_weights()
model.fit(conjunto_treino, rotulo_treino, validation_data=(conjunto_teste, rotulo_teste), epochs=epocas)
pesos_final = model.get_weights()

# validacao
print('**********************************************************************')
resultado = model.predict(conjunto_validacao[:tamanho_amostra_validacao])

numero_acertos = 0
acertos = [False for _ in range(tamanho_amostra_validacao)]
valores_esperados = ['' for _ in range(tamanho_amostra_validacao)]
valores_preditos = ['' for _ in range(tamanho_amostra_validacao)]

verdadeiro_positivo: int = 0
falso_positivo: int = 0
falso_negativo: int = 0
verdadeiro_negativo: int = 0

# Verificacao do resultado
for i in range(0, tamanho_amostra_validacao):
    numero_amostra = str(i + 1).rjust(4, '0')
    print('****************************** ' + numero_amostra + ' ******************************')
    acertos[i], (valores_esperados[i], valores_preditos[i]), matriz = (
        resultados.verifica_resultado(resultado[i], rotulo_validacao[i], dados_mnist))

    print('Acertou: ' + str(acertos[i]))
    print('Predito: ' + str(valores_preditos[i]))
    print('Esperado: ' + str(valores_esperados[i]))

    verdadeiro_positivo += matriz[0]
    falso_positivo += matriz[1]
    falso_negativo += matriz[2]
    verdadeiro_negativo += matriz[3]
    if acertos[i]:
        numero_acertos += 1

percentual_acertos = (numero_acertos / tamanho_amostra_validacao) * 100
print('**********************************************************************')
print('Total de acertos validação: ' + str(numero_acertos))
print('Percentual de acertos validação: ' + str(percentual_acertos) + ' %')
print('')
print('       PREDITO     ')
print('R |-------|-------|')
print('E | ' + str(verdadeiro_positivo).ljust(5) + ' | ' + str(falso_negativo).ljust(5) + ' |')
print('A | ' + str(falso_positivo).ljust(5) + ' | ' + str(verdadeiro_negativo).ljust(5) + ' |')
print('L |-------|-------|')
matriz_confusao = (verdadeiro_positivo, falso_negativo, falso_positivo, verdadeiro_negativo)

dados_utilitario.grava_arquivo_pesos(pesos_inicial, str(time) + '_1_pesos_inicial')
dados_utilitario.grava_arquivo_pesos(pesos_final, str(time) + '_2_pesos_final')
dados_utilitario.grava_arquivo_resultado(acertos, valores_preditos, valores_esperados, numero_acertos, percentual_acertos, matriz_confusao, str(time) + '_3_resultado', tamanho_amostra_validacao)
