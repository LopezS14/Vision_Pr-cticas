import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Cargar la imagen en formato BGR (por defecto en OpenCV)
bgr_image = cv.imread("homer.jpg")
bgr_image2 = cv.imread("fin.jpg")
# Redimensionar bgr_image2 para que coincida con bgr_image
bgr_image2 = cv.resize(bgr_image2, (bgr_image.shape[1], bgr_image.shape[0]))

# Separar los canales imagen 1
B = bgr_image[:, :, 0]  # Canal azul
G = bgr_image[:, :, 1]  # Canal verde
R = bgr_image[:, :, 2]  # Canal rojo
# Separar los canales imagen 2
B1 = bgr_image2[:, :, 0]  # Canal azul
G1 = bgr_image2[:, :, 1]  # Canal verde
R1 = bgr_image2[:, :, 2]  # Canal rojo
#Sumando los canales
Azul=B+B1
Rojo=R+R1
Verde=G+G1
#Negativa
Total=Azul-Rojo-Verde


bgr_image[:,:,0]=R   
bgr_image[:,:,2]=B
# Crear la figura
plt.figure(figsize=(10, 4))

# duplicando la imagen
# Crear otra matriz auxiliar para intercambiar los canales (RGB) canales 1
bgr_aux = np.copy(bgr_image)
bgr_aux[:, :, 0] = R
bgr_aux[:, :,1]  =G
bgr_aux[:, :, 2] = B  
# Crear otra matriz auxiliar para intercambiar los canales (RGB) canales 1
bgr_aux1 = np.copy(bgr_image2)
bgr_aux1[:, :, 0] = R1
bgr_aux1[:, :,1]  =G1
bgr_aux1[:, :, 2] = B1  
#Conversion de RGB a escala de grises
A=(0.299*R)
A2=(0.587*G)
A3=(0.114*B)
AT=(A+A2+A3)
#Conversion a enteros
entero=AT.astype(np.uint8)
#Calculo del histograma para RGB
hist=cv.calcHist(entero,[0],None,[255],[0,255])

#Modificacion de la matriz
C=(100-B)
C2=(100-G)
C3=(100-R)
CT=C+C2+C3
#suma de canales

# Imagen original BGR
plt.subplot(1, 11, 1)
plt.imshow(cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB))  # Convertir BGR a RGB para visualizar bien
plt.imshow(bgr_aux)
plt.axis("off")
plt.title("Imagen en RGB")

# Canal Azul
plt.subplot(1, 11, 2)
plt.imshow(B, cmap="Blues")  # Mostrar en escala de azules
plt.axis("off")
plt.title("Canal Azul")

# Canal Verde
plt.subplot(1, 11, 3)
plt.imshow(G, cmap="Greens")  # Mostrar en escala de verdes
plt.axis("off")
plt.title("Canal Verde")

# Canal Rojo
plt.subplot(1,11, 4)
plt.imshow(R, cmap="Reds")  # Mostrar en escala de rojos
plt.axis("off")
plt.title("Canal Rojo")
#Conversion a escala de grises
plt.subplot(1,2,1)
plt.imshow(AT,cmap='gray')
plt.axis("off")
plt.title("Escala de grises")
#Modificacion
plt.subplot(1,11,6)
plt.imshow(CT)
plt.title("modificada")
plt.axis("off")
#Original 2
plt.subplot(1,11,7)
plt.title("original 2")
plt.imshow(bgr_aux1)
plt.axis("off")
#Canales azules
plt.subplot(1,11,8)
plt.imshow(Azul,cmap='Blues')
plt.title("canales azules")
plt.axis("off")
#Canales verdes
plt.subplot(1,11,9)
plt.imshow(Verde,cmap='Greens')
plt.title("canales Verdes")
plt.axis("off")
#Canales rojos
plt.subplot(1,11,10)
plt.imshow(Rojo,cmap='Reds')
plt.title("Canales rojos")
plt.axis("off")
#Invertido
plt.subplot(1,11,11)
plt.imshow(Total)
plt.axis("off")
plt.title("Invertido")


# Mostrar la imagen
plt.tight_layout()

plt.show()
