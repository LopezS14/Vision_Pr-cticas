from random import randint
import numpy as np
import matplotlib.pyplot as plt

# Función para llenar la matriz con valores aleatorios de 0 a 255
def llenar_matriz(filas, columnas):
    matriz = [[randint(0, 255) for _ in range(columnas)] for _ in range(filas)]
    return np.array(matriz)  # Convertir la matriz a un array de NumPy

# Función para generar matrices de diferentes tamaños
def generar_matrices():
    matrices = []
    for i in range(2, 6):  # Escalando por factores de 2 a 5
        matriz_base = llenar_matriz(3, 3)
        matriz_expandida = np.repeat(np.repeat(matriz_base, i, axis=0), i, axis=1)
        matrices.append(matriz_expandida)
    return matrices

# Generar las matrices
matrices_crecientes = generar_matrices()

# Graficar matrices
plt.figure(figsize=(10, 10))

for graficar, matriz in enumerate(matrices_crecientes, start=1):
    plt.subplot(2, 2, graficar)  # Ajustar para 4 imágenes (2 filas, 2 columnas)
    plt.title(f"matriz {graficar+1}")
    plt.imshow(matriz, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')  # Ocultar ejes

plt.tight_layout()
plt.show()
