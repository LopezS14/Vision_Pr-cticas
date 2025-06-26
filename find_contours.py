import cv2
import matplotlib.pyplot as plt


#Seleccionar la camara
camara = cv2.VideoCapture(0)  # 0 para la camara principal  #2 para la camara con iPhone

#leer el video
ret,frame = camara.read()

#Guardar la imagen
if ret:
    cv2.imwrite("Nombre.jpg",frame)
    print("Imagen guardada")

else:
    print("Error al guardar la imagen")

camara.release()
cv2.destroyAllWindows

#Bordes en la imagen
imagen = cv2.imread('imagenes/monedas1.jpg')
#imagen2 = cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB)
imageng = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)

#Usar canny para evitar el umbralizado y contornos
canny = cv2.Canny(imageng,50,255)
#dilatación
dilatacion = cv2.dilate(canny,None,iterations=11)
#Erosion
erosion = cv2.erode(dilatacion,None,iterations=1)

#Sobel 
sobel = cv2.Sobel(imageng,cv2.CV_64F,1,0,ksize=5)

#Contornos
contornos,_ = cv2.findContours(erosion,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# Iterar sobre los contornos y calcular la longitud
print(len(contornos))
#dibujar los contornos
cv2.drawContours(imagen,contornos,-1,(0,0,0),5)


#Mostrar
plt.subplot(1,3,1)
plt.axis("off")
plt.imshow(imagen)
plt.subplot(1,3,2)
plt.imshow(erosion,cmap='gray')
plt.axis("off")
"""plt.subplot(1,3,3)
plt.imshow(contornos,cmap='gray')"""
plt.show()