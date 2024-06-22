import numpy as np
from keras.api.models import Sequential
from keras.api.layers import Dense, Conv2D, Flatten
import entrada_dados

# importacao dos dados
####################################################################

# conjunto de dados mnist
conjunto_treino, rotulo_treino, conjunto_teste, rotulo_teste, conjunto_validacao, rotulo_validacao = entrada_dados.monta_conjunto_dados_mnist()

model = Sequential()

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(conjunto_treino, rotulo_treino, validation_data=(conjunto_teste, rotulo_teste), epochs=5)

print('********************************************************************')
resultado = model.predict(conjunto_validacao[:1000])
acertos = 0
for i in range(0, 1000):
    print('****************************** ' + str(i) + ' ******************************')
    acerto, valor_esperado, valor_predito = entrada_dados.verifica_resultado_mnist(resultado[i], rotulo_validacao[i])
    print('Acertou: ' + str(acerto))
    print('Predito: ' + str(valor_predito))
    print('Esperado: ' + str(valor_esperado))
    if acerto:
        acertos += 1
percentual_acertos = acertos / 1000
print('********************************************************************')
print('Total de acertos validação: ' + str(acertos))
print('Percentual de acertos validação: ' + str(percentual_acertos * 100) + ' %')
