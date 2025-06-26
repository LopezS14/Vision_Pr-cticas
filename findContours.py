import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar imagen
img = cv2.imread('homer.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Binarizar con Canny (bordes)
edges = cv2.Canny(gray, 100, 200)

# 3. Simular findContours manualmente
def contornos_manuales(binaria):
    contornos = []
    h, w = binaria.shape
    for i in range(1, h-1):
        for j in range(1, w-1):
            if binaria[i, j] == 255:
                vecinos = binaria[i-1:i+2, j-1:j+2]
                if np.any(vecinos == 0):
                    contornos.append((i, j))
    return contornos

# 4. Aplicar función manual
contornos = contornos_manuales(edges)

# 5. Crear una imagen para mostrar los contornos
img_contornos = np.zeros_like(edges)
for i, j in contornos:
    img_contornos[i, j] = 255

# 6. Mostrar resultados
plt.figure(figsize=(10,4))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Binarización de la imagen con canny")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Contornos manuales")
plt.imshow(img_contornos, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
