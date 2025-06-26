import numpy as np
import matplotlib.pyplot as plt

# Primer arreglo del 1 al 5
arreglo_1 = np.array([1, 2, 3, 4, 5]) #Eje x

# Segundo arreglo con valores entre 0.8 y 4
arreglo_2 = np.array([0.8, 1.3, 2, 3, 4]) #eje Y

# Calculamos el promedio de ambos arreglos
promedio_1 = np.mean(arreglo_1)
promedio_2 = np.mean(arreglo_2)

# Mostrar los resultados
print(f"Primer arreglo: {arreglo_1}")
print(f"Promedio del primer arreglo: {promedio_1}")

print(f"Segundo arreglo: {arreglo_2}")
print(f"Promedio del segundo arreglo: {promedio_2}")

# Cálculo de los coeficientes de la recta de regresión
# a1 (pendiente)
numerador = np.sum((arreglo_1 - promedio_1) * (arreglo_2 - promedio_2))
denominador = np.sum((arreglo_1 - promedio_1) ** 2)
a1 = numerador / denominador

# a0 (intersección)
a0 = promedio_2 - a1 * promedio_1
print(f"a0: {a0}")
print(f"a1: {a1}")

# Ecuación de la recta de regresión: y = a1 * x + a0
linea_regresion = a1 * arreglo_1 + a0

# Graficando los resultados
plt.scatter(arreglo_1, arreglo_2, color='blue', marker='o', label='Puntos de datos')

# Dibujando la línea de regresión
plt.plot(arreglo_1, linea_regresion, color='red', label='Línea de regresión')

# Añadir etiquetas y título
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.title('Gráfico de dispersión con regresión lineal')

# Mostrar la leyenda
plt.legend()

# Mostrar el gráfico
plt.show()

# Mostrar los coeficientes de la recta de regresión
print(f"Coeficiente a1 (pendiente): {a1}")
print(f"Coeficiente a0 (intersección): {a0}")
