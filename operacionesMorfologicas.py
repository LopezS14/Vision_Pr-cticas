import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Leer la imagen
image_path = "homer.jpg"  
image = cv.imread(image_path)

# Convertir a escala de grises
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Umbralizar la imagen
_, binary = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)

# Aplicar filtro Canny
filtroCanny = cv.Canny(binary, 100, 200)

# Operación morfológica (dilatación)
kernel = np.ones((5,5), np.uint8)
dilatacion= cv.dilate(filtroCanny, kernel, iterations=5)
#operacion morfologica (Erosion)
erosion=cv.erode(filtroCanny,kernel,iterations=1)
#Aplicar cierre
Aplicar1=cv.dilate(image,kernel,iterations=1)
erosion1=cv.erode(image,kernel,iterations=1)
Cierre=Aplicar1+erosion1

#Aplicar Apertura
Aplicar=cv.erode(image,kernel,iterations=1)
dilatar=cv.dilate(image,kernel,iterations=1)
Apertura=Aplicar+dilatar

#Segmentar para colores en hsv
# Convertir a espacio de color HSV
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# Definir los rangos de color
#Rango de amarillo
bajo = np.array([25, 100,100])
alto = np.array([35, 255, 255])

 # Generar la máscara
mask = cv.inRange(hsv, bajo, alto)
#dilatar la mascara
mask2=cv.dilate(mask,kernel,iterations=5)
#Segementar por el color amarillo
segmentar=cv.bitwise_and(image,image,mask=mask)

# Graficar
plt.subplot(1, 7, 1)
plt.imshow(binary, cmap='gray')
plt.title("Rgb a bgr")
plt.axis("off")

plt.subplot(1, 7, 2)
plt.imshow(filtroCanny, cmap='gray')
plt.title("Canny")
plt.axis("off")

plt.subplot(1,8, 3)
plt.imshow(dilatacion, cmap='gray')
plt.title("Dilata")
plt.axis("off")
plt.subplot(1,8,4)
plt.imshow(erosion,cmap='gray')
plt.title("erosion")
plt.axis("off")
plt.subplot(1,8,5)
plt.imshow(Apertura,cmap='Accent')
plt.title("aper")
plt.axis("off")
plt.subplot(1,8,6)
plt.imshow(Cierre,cmap='Accent')
plt.title("cierre")
plt.axis("off")
plt.subplot(1,8,7)
plt.imshow(segmentar,cmap='Accent')
plt.title("bit")
plt.axis("off")
plt.subplot(1,8,8)
plt.imshow(mask2,cmap='Accent')
plt.title("dilatar_mak")

plt.axis("off")

plt.tight_layout()

plt.show()
