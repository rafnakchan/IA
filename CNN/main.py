from keras.api.models import Sequential
from keras.api.layers import Dense, Conv2D, Flatten
import entrada_dados

# importacao dos dados
####################################################################

# conjunto de dados fornecidos pela Sarajane
conjunto_treino, conjunto_teste, conjunto_validacao = entrada_dados.monta_conjunto_dados('./Arquivos/X.txt')
rotulo_treino, rotulo_teste, rotulo_validacao = entrada_dados.monta_rotulo('./Arquivos/Y_letra.txt')

model = Sequential()

model.add(Conv2D(60, kernel_size=3, activation='relu', input_shape=(10, 12, 1)))
model.add(Conv2D(30, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(26, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(conjunto_treino, rotulo_treino, validation_data=(conjunto_teste, rotulo_teste), epochs=50)

print('**********************************************************************')
resultado = model.predict(conjunto_validacao[:130])
acertos = 0
for i in range(0, 130):
    print('****************************** ' + str(i) + ' ******************************')
    acerto, valor_esperado, valor_predito = entrada_dados.verifica_resultado(resultado[i], rotulo_validacao[i])
    print('Acertou: ' + str(acerto))
    print('Predito: ' + str(valor_predito))
    print('Esperado: ' + str(valor_esperado))
    if acerto:
        acertos += 1
percentual_acertos = acertos / 130
print('**********************************************************************')
print('Total de acertos validação: ' + str(acertos))
print('Percentual de acertos validação: ' + str(percentual_acertos * 100) + ' %')
