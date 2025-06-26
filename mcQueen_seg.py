import cv2
import numpy as np
import matplotlib.pyplot as plt

# Leer la imagen
img = cv2.imread("imagenes/rayo_ruido.jpg")

# Preprocesamiento: reducción de ruido sal y pimienta
mediana = cv2.medianBlur(img, 5)
gauss = cv2.GaussianBlur(mediana, (9, 9), 0)

# Convertir imagen preprocesada a HSV
hsv_img = cv2.cvtColor(gauss, cv2.COLOR_BGR2HSV)

# === Definir rangos HSV ===
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])

lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])

lower_blue = np.array([100, 100, 100])
upper_blue = np.array([130, 255, 255])

lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])

# === Crear máscaras ===
mask_red1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)

mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
mask_white = cv2.inRange(hsv_img, lower_white, upper_white)
mask_blue = cv2.inRange(hsv_img, lower_blue, upper_blue)
mask_black = cv2.inRange(hsv_img, lower_black, upper_black)

# === Sumar máscaras de interés (por ejemplo, rojo + amarillo) ===
mask_sum = cv2.bitwise_or(mask_red, mask_yellow, mask_white)
mask_sum = cv2.bitwise_or(mask_sum, mask_white)

# Limpiar máscara (morfología)
kernel = np.ones((5, 5), np.uint8)
mask_clean = cv2.morphologyEx(mask_sum, cv2.MORPH_CLOSE, kernel, iterations=2)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)

# === Contornos ===
contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask_final = np.zeros_like(mask_clean)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 300:  
        cv2.drawContours(mask_final, [cnt], -1, 255, thickness=cv2.FILLED)

# === Aplicar la máscara ===
segmented = cv2.bitwise_and(img, img, mask=mask_final)
segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)

# === Mostrar resultados ===
plt.figure(figsize=(10, 5))
plt.imshow(segmented_rgb)
plt.title("Rayo McQueen segmentado")
plt.axis("off")
plt.show()
