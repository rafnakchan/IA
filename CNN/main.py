import time

import numpy as np
from keras.api.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.api.models import Sequential

import dados_utilitario

# importacao dos dados
####################################################################

# Mudar para False se desejar usar os dados do EP 1 do MLP
dados_mnist: bool = True
conjunto_treino, rotulo_treino, conjunto_teste, rotulo_teste, conjunto_validacao, rotulo_validacao = (
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

model.add(Conv2D(filtros_primeira_camada, kernel_size=tamanho_kernel, activation='relu', input_shape=formato_imagem))
model.add(MaxPool2D(pool_size=tamanho_pooling))
model.add(Conv2D(filtros_segunda_camada, kernel_size=tamanho_kernel, activation='relu'))
model.add(MaxPool2D(pool_size=tamanho_pooling))
model.add(Flatten())
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
for i in range(0, tamanho_amostra_validacao):
    print('****************************** ' + str(i + 1) + ' ******************************')
    acertos[i], valores_esperados[i], valores_preditos[i] = dados_utilitario.verifica_resultado(resultado[i], rotulo_validacao[i], dados_mnist)
    print('Acertou: ' + str(acertos[i]))
    print('Predito: ' + str(valores_preditos[i]))
    print('Esperado: ' + str(valores_esperados[i]))
    if acertos[i]:
        numero_acertos += 1

percentual_acertos = numero_acertos / tamanho_amostra_validacao
print('**********************************************************************')
print('Total de acertos validação: ' + str(numero_acertos))
print('Percentual de acertos validação: ' + str(percentual_acertos * 100) + ' %')

dados_utilitario.grava_arquivo_pesos(pesos_inicial, str(time) + '_1_pesos_inicial')
dados_utilitario.grava_arquivo_pesos(pesos_final, str(time) + '_2_pesos_final')
dados_utilitario.grava_arquivo_resultado(acertos, valores_preditos, valores_esperados, str(time) + '_3_resultado', tamanho_amostra_validacao)
