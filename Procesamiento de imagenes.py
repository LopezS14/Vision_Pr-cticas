import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

#Aplicar imagen
image_path = "imagenes/rayo_ruido.jpg"  
image = cv.imread(image_path)
# Aplicar filtro Gaussiano
Eliminar = cv.GaussianBlur(image, (9,9), 0)

# Aplicar filtro de mediana para reducir ruido sal y pimienta
MedianaFilter = cv.medianBlur(image, 5)
MF2 = cv.medianBlur(Eliminar,5)  # Aplicado sobre la imagen suavizada

#Deteccion de bordes
# Convertir la imagen a escala de grises
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

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

# Convertir a HSV
# Convertir a HSV
hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# Definir rangos de rojo correctamente
lower_red1 = np.array([0, 100, 100])   # Primer rango (rojo bajo)
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 100, 100])  # Segundo rango (rojo alto)
upper_red2 = np.array([180, 255, 255])
#Definir los rangos de negrop

# Crear máscaras para ambos rangos
mask1 = cv.inRange(hsv_img, lower_red1, upper_red1)
mask2 = cv.inRange(hsv_img, lower_red2, upper_red2)

# Combinar las máscaras correctamente con bitwise_or
red_mask = cv.bitwise_or(mask1, mask2)

# Aplicar la máscara a la imagen original
result = cv.bitwise_and(image, image, mask=red_mask)


# Graficar resultados
plt.figure(figsize=(12, 4))

plt.subplot(1,5,1)
plt.title("Original")
plt.axis("off")
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))  # Convertir BGR a RGB para mostrar correctamente

plt.subplot(1,5,2)
plt.title("Eliminar ruido Gaussiano")
plt.axis("off")
plt.imshow(cv.cvtColor(Eliminar, cv.COLOR_BGR2RGB))

plt.subplot(1,5,3)
plt.title("Quitar sal y pimienta")
plt.axis("off")
plt.imshow(cv.cvtColor(MF2, cv.COLOR_BGR2RGB))

plt.subplot(1,5,4)
plt.title("Imagen en HSV")
plt.axis("off")
plt.imshow(result,cmap='Accent')
plt.subplot(1,5,5)
plt.title("deteccion de bordes")
plt.imshow(sobel_horizontal,cmap='gray')
plt.axis("off")

plt.show()
