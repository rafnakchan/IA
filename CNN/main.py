import numpy as np
from keras.api.models import Sequential
from keras.api.layers import Dense, Conv2D, Flatten
import dados_utilitario

# importacao dos dados
####################################################################

dados_mnist: bool = True
if dados_mnist:
    # conjunto de dados mnist
    conjunto_treino, rotulo_treino, conjunto_teste, rotulo_teste, conjunto_validacao, rotulo_validacao = (
        dados_utilitario.monta_conjunto_dados_mnist())
else:
    # conjunto de dados fornecidos pela Sarajane
    conjunto_treino, rotulo_treino, conjunto_teste, rotulo_teste, conjunto_validacao, rotulo_validacao = (
        dados_utilitario.monta_conjunto_dados('./Arquivos/X.txt', './Arquivos/Y_letra.txt'))

# configuracoes
epocas: int = 5
tamanho_amostra_validacao: int = np.size(conjunto_validacao, axis=0)
neuronios_primeira_camada: int = 64
neuronios_segunda_camada: int = 32
tamanho_kernel: int = 3
formato_imagem: tuple[int, int, int] = (np.size(conjunto_treino, axis=1), np.size(conjunto_treino, axis=2), 1)
neuronios_camada_saida: int = np.size(rotulo_treino, axis=1)

# montagem do modelo
model = Sequential()

model.add(Conv2D(neuronios_primeira_camada, kernel_size=tamanho_kernel, activation='relu', input_shape=formato_imagem))
model.add(Conv2D(neuronios_segunda_camada, kernel_size=tamanho_kernel, activation='relu'))
model.add(Flatten())
model.add(Dense(neuronios_camada_saida, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# treinamento
model.fit(conjunto_treino, rotulo_treino, validation_data=(conjunto_teste, rotulo_teste), epochs=epocas)

# validacao
print('**********************************************************************')
resultado = model.predict(conjunto_validacao[:tamanho_amostra_validacao])
acertos = 0
for i in range(0, tamanho_amostra_validacao):
    print('****************************** ' + str(i + 1) + ' ******************************')
    if dados_mnist:
        acerto, valor_esperado, valor_predito = dados_utilitario.verifica_resultado_mnist(resultado[i], rotulo_validacao[i])
    else:
        acerto, valor_esperado, valor_predito = dados_utilitario.verifica_resultado(resultado[i], rotulo_validacao[i])
    print('Acertou: ' + str(acerto))
    print('Predito: ' + str(valor_predito))
    print('Esperado: ' + str(valor_esperado))
    if acerto:
        acertos += 1
percentual_acertos = acertos / tamanho_amostra_validacao
print('**********************************************************************')
print('Total de acertos validação: ' + str(acertos))
print('Percentual de acertos validação: ' + str(percentual_acertos * 100) + ' %')
