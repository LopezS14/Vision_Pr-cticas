import numpy as np
import matplotlib.pyplot as plt
import cv2  as cv
import math

image_path = "imagen.jpg"  # Cambia el nombre de archivo según tu imagen
image = cv.imread(image_path)
#definicion de funciones
#ruido gaussiano
def add_gaussian_noise(image, mean=0, sigma=100):

    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    "todos los valores que sobrepasen al 255 sea 255"
    noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy_image
#ruido de sal y pimienta
"A diferencia de el gaussiano, no existe una funcion como tal, se le agregan pixeles blancos y pixeles negros"
"se copia la imagen para poder agregar informacion"
def add_salt_and_pepper_noise(image, salt_prob=0.5, pepper_prob=0.5):

    row, col, ch = image.shape
    noisy_image = image.copy()

    # Salt (valor máximo)
    num_salt = int(salt_prob * row * col)
    salt_coords = [np.random.randint(0, i-1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1], :] = 255

    # Pepper (valor mínimo)
    num_pepper = int(pepper_prob * row * col)
    pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1], :] = 0

    return noisy_image

#ruido mixto

def add_mixed_noise(image, mean=0, sigma=100, salt_prob=0.01, pepper_prob=0.01):
    gaussian_noisy_image = add_gaussian_noise(image, mean, sigma)
    mixed_noisy_image = add_salt_and_pepper_noise(gaussian_noisy_image, salt_prob, pepper_prob)
    return mixed_noisy_image
#filtrar los ruidos 


#verificar si se carga la imagen
if image is None:
    print(f"No se pudo cargar la imagen en {image_path}")
else:
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Convertir de BGR a RGB pasarlo a la funcion

    # Aplicar ruidos
    gaussian_noisy_image = add_gaussian_noise(image)
    salt_pepper_noisy_image = add_salt_and_pepper_noise(image)
    mixed_noised=add_mixed_noise(image)
    #Eliminar ruidos gaussiano
    "Kernel es un arreglo que recorre toda la imagen, es una convolucion"
    "Matrices menores en el kernel, verificar la convolucion de la imagen"
    "filtro de suavizado de imagen pierdes informacion"
    Eliminar=cv.GaussianBlur(gaussian_noisy_image,(9,9),0)
    #Aplicar filtro de mediana
    "trabaja sobre toda una comunidad de pixeles"
    MedianaFilter=cv.medianBlur(salt_pepper_noisy_image,5)
    #Aplicar al filtro se sal y pimienta al gaussiano
    MF2=cv.medianBlur(gaussian_noisy_image,5)
    #Umbralizar  
    # Conversión a enteros
    entero = cv.cvtColor(MedianaFilter, cv.COLOR_RGB2GRAY)

    # Umbralización
    m, n = np.shape(entero)
    for i in range(m):
        for j in range(n):
            if entero[i, j] < 100:
                entero[i, j] = 0
            else:
                entero[i, j] = 1

        #Aplicar el filtro sobel a esta
    sobel = cv.Sobel(gaussian_noisy_image, cv.CV_64F, 1, 0, ksize=3)

    #Aplicar el gradiente
    #calcular la magnitud de la gradiante
    # Aplicar el filtro Sobel
    sobel_horizontal = cv.Sobel(sobel, cv.CV_64F, 1, 0, ksize=3)
    sobel_vertical = cv.Sobel(sobel, cv.CV_64F, 0, 1, ksize=3)
    Gradiante=np.sqrt(sobel_horizontal**2 + sobel_vertical**2)
    

    # Mostrar imágenes
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 8, 1)
    plt.imshow(image)
    plt.title("Imagen Original")
    plt.axis("off")

    plt.subplot(1, 8, 2)
    plt.imshow(gaussian_noisy_image,cmap="Accent")
    plt.title("Gaussiano")
    plt.axis("off")

    plt.subplot(1, 8, 3)
    plt.imshow(salt_pepper_noisy_image,cmap="Accent")
    plt.title("SaltPeper")
    plt.axis("off")

    plt.subplot(1,8,4)
    plt.imshow(Eliminar,cmap='Accent')
    plt.title("filGauss")
    plt.axis("off")

    plt.subplot(1,8,5)
    plt.imshow(MedianaFilter,cmap="Accent")
    plt.title("FiltroSyP")
    plt.axis("off")
    plt.subplot(1,8,6)
    plt.imshow(MF2,cmap="Accent")
    plt.title("Fil-Gaussiano")
    plt.axis("off")
    plt.tight_layout()
    plt.subplot(1,8,7)
    plt.imshow(mixed_noised,cmap="Accent")
    plt.title("Rui_Mixto")
    plt.axis("off")
    plt.subplot(1,8,8)
    plt.imshow(entero,cmap='gray')
    plt.axis("off")
    plt.tight_layout()

    plt.show()
