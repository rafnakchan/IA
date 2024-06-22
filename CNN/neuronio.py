import numpy as np
import funcao_ativacao


class Neuronio:

    def __init__(self, numero_entradas: int):
        self.numero_entradas = numero_entradas
        self.pesos = np.random.uniform(-0.1, 0.1, numero_entradas)
        self.bias = np.random.uniform(-0.1, 0.1)

    def print_pesos(self):
        print(self.pesos)

    # soma todos os pesos multiplicados pela entrada do neuronio
    def somatoria_entradas(self, entradas):
        somatorio = 0.0
        for i in range(0, self.numero_entradas):
            somatorio += (entradas[i] * self.pesos[i])
        return somatorio + self.bias

    # retorna o resultado da funcao de ativacao
    def ativacao(self, entradas):
        somatoria_entradas = self.somatoria_entradas(entradas)
        return funcao_ativacao.sigmoide(somatoria_entradas)

    def atualizar_pesos(self, correcao_pesos):
        for i in range(0, self.numero_entradas):
            self.pesos[i] += correcao_pesos[i]
        self.bias += correcao_pesos[self.numero_entradas]
