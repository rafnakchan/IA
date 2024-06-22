import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Dense, Conv2D, Flatten
import entrada_dados

# importacao dos dados
####################################################################

# configuração para imprimir o array completo
np.set_printoptions(threshold=np.inf)

# remodelando os dados que ficaram em 1 dimensao para uma matriz
conjunto = entrada_dados.monta_conjunto_dados('./Arquivos/X.txt')
rotulo = entrada_dados.monta_rotulo('./Arquivos/Y_letra.txt')

conjunto_treino, conjunto_resto = train_test_split(conjunto, train_size=1066, random_state=42, shuffle=False)
rotulo_treino, rotulo_resto = train_test_split(rotulo, train_size=1066, random_state=42, shuffle=False)

conjunto_teste, conjunto_validacao = train_test_split(conjunto_resto, test_size=0.5, random_state=42, shuffle=False)
rotulo_teste, rotulo_validacao = train_test_split(rotulo_resto, test_size=0.5, random_state=42, shuffle=False)

model = Sequential()

model.add(Conv2D(60, kernel_size=3, activation='relu', input_shape=(10, 12, 1)))
model.add(Conv2D(30, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(26, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(conjunto_treino, rotulo_treino, validation_data=(conjunto_teste, rotulo_teste), epochs=50)

acertos = 0
for i in range(0, 130):
    resultado = model.predict(conjunto_validacao[i:i + 1])
    acerto, letra_esperada, letra_predita = entrada_dados.verifica_resultado(resultado[0], rotulo_validacao[i])
    print('Acertou: ' + str(acerto))
    print('Letra predita: ' + str(letra_predita))
    print('Letra esperada: ' + str(letra_esperada))
    if acerto:
        acertos += 1
    # plt.imshow(conjunto_validacao[i], origin='upper', vmin=0.0, vmax=1.0)
    # plt.colorbar()
    # plt.show()
percentual_acertos = acertos / 130
print('Total de acertos validação: ' + str(acertos))
print('Percentual de acertos validação: ' + str(percentual_acertos))
