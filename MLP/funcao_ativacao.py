import numpy as np


# funcao de ativacao sigmoide
def sigmoide(x: float):
    return 1 / (1 + np.exp(-x))


# derivada da funcao sigmoide
def derivada_sigmoide(x: float):
    y = sigmoide(x)
    return y * (1 - y)
