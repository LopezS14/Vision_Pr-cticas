import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math

# Cargar la imagen
img_path = cv.imread('homer.jpg')
hsv_img = cv.cvtColor(img_path, cv.COLOR_BGR2HSV)

# Canales de la imagen
R = img_path[:, :, 0] 
G = img_path[:, :, 1]
B = img_path[:, :, 2]

# Conversión de RGB al multiplicar
A = (5 * R)
A2 = (5 * G)
A3 = (5 * B)
AT = (A + A2 + A3)
#Conversion al sumar
K=(5+R)
K2=(5+G)
K3=(5+B)
KT=(K+K2+K3)
#Conversion de hsv
T = (5 * R)
T2 = (5 * G)
T3 = (5 * B)
TT = (A + A2 + A3)

# Conversión a enteros
entero = AT.astype(np.uint8)
entero2=KT.astype(np.uint8)
entero3=TT.astype(np.uint8)

plt.figure(1)
plt.subplot(1,3,1)
plt.title("Modificando los valores Mult")
plt.imshow(entero)
plt.axis("off")
plt.subplot(1,3,2)

plt.title("Modificando SUMA")
plt.imshow(entero2)
plt.axis("off")
plt.subplot(1,3,3)
plt.title("original a HSV")
plt.imshow(entero3)
plt.axis("off")
plt.tight_layout()

plt.show()