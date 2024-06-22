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

conjunto_treino, resto = train_test_split(conjunto, train_size=1066, random_state=42, shuffle=False)

conjunto_teste, conjunto_validacao = train_test_split(resto, test_size=0.5, random_state=42, shuffle=False)

plt.imshow(conjunto_treino[0], origin='upper', vmin=-1.0, vmax=1.0)
plt.colorbar()
plt.show()

model = Sequential()

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(10, 12, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(26, activation='softmax'))

for amostra in conjunto:
    amostra_expandida = np.expand_dims(amostra, axis=2)
    print(amostra_expandida.shape)
    print(amostra_expandida[0])
