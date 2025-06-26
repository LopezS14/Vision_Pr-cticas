import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img_path = cv.imread('homer.jpg')


# Canales
B = img_path[:,:,0]
G = img_path[:,:,1]
R = img_path[:,:,2]

def rgb_to_hsv(R, G, B):
    R, G, B = R / 255.0, G / 255.0, B / 255.0
    mx = max(R, G, B)
    mn = min(R, G, B)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == R:
        h = (60 * ((G - B) / df) + 360) % 360
    elif mx == G:
        h = (60 * ((B - R) / df) + 120) % 360
    elif mx == B:
        h = (60 * ((R - G) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df / mx) * 100
    v = mx * 100
    return h, s, v

# Convertir toda la imagen de RGB a HSV
hsv_img = np.zeros_like(img_path, dtype=np.float32)

for i in range(img_path.shape[0]):
    for j in range(img_path.shape[1]):
        B, G, R = img_path[i, j]    # <- Cambié el orden
        h, s, v = rgb_to_hsv(R, G, B)
        hsv_img[i, j] = [h, s, v]


# Función para convertir HSV (tu rango) a RGB (0-1)
def hsv_to_rgb(h, s, v):
    s /= 100.0
    v /= 100.0
    c = v * s
    h_prime = h / 60.0
    x = c * (1 - abs(h_prime % 2 - 1))

    r = g = b = 0

    if 0 <= h_prime < 1:
        r, g, b = c, x, 0
    elif 1 <= h_prime < 2:
        r, g, b = x, c, 0
    elif 2 <= h_prime < 3:
        r, g, b = 0, c, x
    elif 3 <= h_prime < 4:
        r, g, b = 0, x, c
    elif 4 <= h_prime < 5:
        r, g, b = x, 0, c
    elif 5 <= h_prime < 6:
        r, g, b = c, 0, x

    m = v - c
    r += m
    g += m
    b += m

    return r, g, b

# Crear imagen RGB para mostrar (valores de 0 a 1)
rgb_img = np.zeros_like(hsv_img, dtype=np.float32)

for i in range(hsv_img.shape[0]):
    for j in range(hsv_img.shape[1]):
        h, s, v = hsv_img[i, j]
        r, g, b = hsv_to_rgb(h, s, v)
        rgb_img[i, j] = [r, g, b]

# Mostrar la imagen convertida de HSV a RGB
plt.figure(figsize=(10, 5))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(img_path)
plt.axis('off')
plt.subplot(1,3,2)
plt.title("HSV")
plt.imshow(hsv_img)
plt.axis ('off')
plt.subplot(1,3,3)
plt.imshow(rgb_img)
plt.title("RGB")
plt.axis('off')
plt.show()


