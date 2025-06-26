import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Cargar la imagen
img_path = cv.imread('fin.jpg')

# Convertir a espacio de color HSV
hsv = cv.cvtColor(img_path, cv.COLOR_BGR2HSV)

# Definir los rangos de color en formato NumPy
bajo = np.array([0, 50, 50])
alto = np.array([10, 255, 255])

 # Generar la máscara
mask = cv.inRange(hsv, bajo, alto)
#Segementar por el color azul
segmentar=cv.bitwise_and(img_path,img_path,mask=mask)
#conversion de bgr a RGB
destRGB = cv.cvtColor(segmentar, cv.COLOR_BGR2RGB)
 
# Mostrar la máscara
plt.subplot(1,4,1)
plt.imshow(mask, cmap='gray')  # Usar 'gray' en lugar de 'hsv' para visualizar mejor la máscara
plt.title("Máscara")
plt.axis("off")
plt.subplot(1,4,2)
plt.imshow(segmentar,cmap='hsv')
plt.title("Segmentacion")
plt.axis("off")
plt.subplot(1,4,3)
plt.imshow (img_path,cmap='Accent')
plt.axis("off")
plt.title("original")
plt.subplot(1,4,4)
plt.title("BRG to RGB")
plt.imshow(destRGB,cmap='gray')
plt.axis("off")

plt.show()
