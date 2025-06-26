#Importacion de librerias{
import cv2 as cv
import matplotlib.pyplot as plt 
import numpy as np
img_path=cv.imread('homer.jpg')
#canales
R=img_path[:,:,0]
G=img_path[:,:,1]
B=img_path[:,:,2]
#Creacion de la matriz para guardar los valores
matriz=np.empty_like(img_path)
#Iniciando conversion a YCRB matriz
"Luminancia que cada color aporta [ .299,.587,.114]"
YC=matriz[:,:,0]= .299 *R + .587 * G + .114 * B
"Diferencia entre el azul y el valor de la luminancia"
"[128,-169,-331,-.5]"
CB=matriz[:,:,1]= 128 - .169 * R - .331 * G + .5 * B
"Diferencia entre el canal rojo y el canal de la luminancia"
"[128,.5,.419,.081]"
CR=matriz[:,:,2]= 128+.5 *R-.419*G-.081*B
# sumando los canales para la conversion
sumaCanales=YC+CB+CR

#graficando los resultados de la conversion de los canles
#.figure(figsize=20)
plt.imshow(sumaCanales)
plt.title("Conversion de RGB a YCRB")
plt.axis("off")
plt.show()



