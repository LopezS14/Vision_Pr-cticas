import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math

# Cargar la imagen
img_path = cv.imread('imagen.jpg')

# Canales de la imagen
R = img_path[:, :, 0]
G = img_path[:, :, 1]
B = img_path[:, :, 2]

# Conversión de RGB a escala de grises
A = (0.299 * R)
A2 = (0.587 * G)
A3 = (0.114 * B)
AT = (A + A2 + A3)

# Conversión a enteros
entero = AT.astype(np.uint8)

# Umbralización
m, n = np.shape(entero)
for i in range(m):
    for j in range(n):
        if entero[i, j] < 100:
            entero[i, j] = 0
        else:
            entero[i, j] = 1

# Calcular el histograma
hist = cv.calcHist([entero.astype(np.uint8)], [0], None, [256], [0, 256])

# Rotación de la imagen
rotation45 = 45
# Convertir a radianes
Radianes = rotation45 * np.pi / 180.0

# Obtener dimensiones de la imagen
height, width = img_path.shape[:2]  # Solo tomamos height y width
num_canales = img_path.shape[2] if len(img_path.shape) == 3 else 1

# Crear una imagen de salida con el tamaño adecuado
'Eleva la altura y la anchura al cuadrado, suma esos valores'
'toma la raíz cuadrada del resultado y luego convierte el valor a un número entero.'
max_len = int(math.sqrt(height**2 + width**2))
rotated_image = np.zeros((max_len, max_len, num_canales), dtype=np.uint8)

rotated_height, rotated_width = rotated_image.shape[:2]
mid_row = (rotated_height) / 2
mid_col = (rotated_width )/ 2

# Rotación manual de la imagen
for r in range(rotated_height):
    for c in range(rotated_width):
        # Aplicar matriz de rotación inversa
        y = (r - mid_col) * math.cos(Radianes) + (c - mid_row) * math.sin(Radianes)
        x = -(r - mid_col) * math.sin(Radianes) + (c - mid_row) * math.cos(Radianes)

        # Ajustar valores
        y += mid_col
        x += mid_row

        # Redondear valores
        x = round(x)
        y = round(y)

        # Verificar si están dentro de la imagen original
        if 0 <= x < width and 0 <= y < height:
            rotated_image[r, c] = img_path[y, x]

# Mostrar resultados
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(hist)
plt.title("Histograma de la imagen binarizada")
plt.xlabel("Valor de píxel")
plt.ylabel("Frecuencia")

plt.subplot(1, 3, 2)
plt.title("Imagen transpuesta")
plt.imshow(entero, cmap='gray')

plt.subplot(1, 3, 3)
plt.imshow(rotated_image, cmap='gray')
plt.title("Imagen rotada")

plt.show()
