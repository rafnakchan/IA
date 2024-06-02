import numpy as np


# funcao de ativacao sigmoide
def sigmoide(x: float):
    if (x <= -709):
        # overflow quando chega + ou - na epoca 50
        # retornando o valor da funcao para x == 709, muito proximo a zero
        return 1.216780750623423e-308
    return 1 / (1 + np.exp(-x))


# derivada da funcao sigmoide
def derivada_sigmoide(x: float):
    y = sigmoide(x)
    return y * (1 - y)
