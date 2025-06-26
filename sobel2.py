import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Leer la imagen
img = cv.imread('homer.jpg')

# Convertir la imagen a escala de grises
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Filtro Prewitt en X y Y
kernel_prewitt_x = np.array([[-1, 0, 1], 
                              [-1, 0, 1], 
                              [-1, 0, 1]])

kernel_prewitt_y = np.array([[-1, -1, -1], 
                              [0,  0,  0], 
                              [1,  1,  1]])

img_FilterX = cv.filter2D(gray, -1, kernel_prewitt_x)
img_FilterY = cv.filter2D(gray, -1, kernel_prewitt_y)
filtros = img_FilterX + img_FilterY

# Aplicar el filtro Sobel
sobel_horizontal = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
sobel_vertical = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

# Calcular la magnitud del gradiente
Gradiante = np.sqrt(sobel_horizontal**2 + sobel_vertical**2)

# Evitar divisiones por 0 en la inversa
Inversa = np.where(Gradiante != 0, 1 / Gradiante, 0)

# Restando y normalizando la imagen resultante
Valor = np.clip(Gradiante - 255, 0, 255)

# Mostrar las imágenes
plt.figure(figsize=(12, 4))
plt.subplot(1, 8, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # Convertir BGR a RGB para visualización correcta
plt.title("Original")
plt.axis("off")

plt.subplot(1, 8, 2)
plt.imshow(gray, cmap='gray')
plt.title("Escala de grises")
plt.axis("off")

plt.subplot(1, 8, 3)
plt.imshow(sobel_horizontal, cmap='gray')
plt.title("Eje X")
plt.axis("off")

plt.subplot(1, 8, 4)
plt.imshow(sobel_vertical, cmap='gray')
plt.title("Eje Y")
plt.axis("off")

plt.subplot(1, 8, 5)
plt.imshow(Gradiante, cmap='gray')
plt.title("Gradiente")
plt.axis("off")

plt.subplot(1, 8, 6)
plt.imshow(Inversa, cmap='gray')
plt.title("Inversa")
plt.axis("off")

plt.subplot(1, 8, 7)
plt.imshow(Valor, cmap='gray')
plt.title("Filtro")
plt.axis("off")

plt.subplot(1, 8, 8)
plt.imshow(filtros, cmap='gray')
plt.title("Prewitt")
plt.axis("off")

plt.tight_layout()
plt.show()
