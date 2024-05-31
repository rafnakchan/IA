import numpy as np


# funcao de ativacao sigmoide
def sigmoide(x):
    if (x <= -709):
        #overflow, checar
        return 1.216780750623423e-308
    resultado = 1 / (1 + np.exp(-x))

    return resultado


# derivada da funcao sigmoide
def derivada_sigmoide(x: float):
    y = sigmoide(x)
    return y * (1 - y)
