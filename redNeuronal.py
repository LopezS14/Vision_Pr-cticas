import numpy as np
import matplotlib.pyplot as plt

# Función sigmoide
def sigmoide(h):
    return 1 / (1 + np.exp(-h))

# Datos
X = np.array([0.5, 1.7, 0.9])
np.random.seed(40)
W = np.random.rand(3)
T = 0.65  # valor esperado (target)
Bias = 1
alpha = 0.001  # tasa de aprendizaje
errores = []  # para graficar
deltaW = np.zeros(3)

# Entrenamiento por épocas
for epoca in range(30):
    h = np.dot(W, X) + Bias
    y = sigmoide(h)

    # Cálculo del error y ajuste con la regla delta
    error = (-(T - y)*(y)*(1-y))
    deltaW = deltaW + ((error)*(X))
    W = W + alpha * deltaW
    print(y)
    errores.append(abs(error))  # guardar error absoluto

# Mostrar error por época
plt.plot(errores)
plt.xlabel("Época")
plt.ylabel("Error Absoluto")
plt.title("Evolución del error")
plt.grid()
plt.show()
