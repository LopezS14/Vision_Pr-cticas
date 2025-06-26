import cv2
import numpy as np

# Paso 1: Capturar imagen desde la cámara
cap = cv2.VideoCapture(0)
print("Presiona 's' para capturar la imagen...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al acceder a la cámara.")
        break

    cv2.imshow("Vista en vivo", frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("captura.jpg", frame)
        break

cap.release()
cv2.destroyAllWindows()

# Paso 2: Leer imagen capturada
image = cv2.imread("captura.jpg")
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

# Paso 3: Detectar contornos
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Mostrar contornos detectados para depuración
debug_contours = image.copy()
cv2.drawContours(debug_contours, contours, -1, (0, 255, 0), 2)
cv2.imshow("Contornos detectados", debug_contours)
cv2.waitKey(0)

# Paso 4: Tomar el contorno más grande (moneda de referencia)
if contours:
    # Detectar círculo envolvente
    (x, y), radius = cv2.minEnclosingCircle(contours[0])
    center = (int(x), int(y))
    radius = int(radius)
    
    if radius > 5:
        cv2.circle(image, center, radius, (0, 255, 0), 2)
        cv2.putText(image, "Referencia", (center[0]-40, center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Paso 5: Calcular relación píxeles/cm
        real_diameter_cm = 2.8  # diámetro real de la moneda
        pixels_per_cm = (radius * 2) / real_diameter_cm
        print(f"[INFO] 1 cm equivale a {pixels_per_cm:.2f} píxeles")

        # Paso 6: Medir otro objeto rectangular
        if len(contours) > 1:
            # Segundo contorno más grande
            rect = cv2.minAreaRect(contours[1])
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

            # Ancho y alto en píxeles
            width_px = min(rect[1])
            height_px = max(rect[1])

            width_cm = width_px / pixels_per_cm
            height_cm = height_px / pixels_per_cm

            print(f"[INFO] Objeto nuevo: {width_cm:.2f} cm de ancho x {height_cm:.2f} cm de alto")

            # Mostrar texto en la imagen
            cv2.putText(image, f"{width_cm:.2f} x {height_cm:.2f} cm", (int(rect[0][0]-50), int(rect[0][1]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            print("No se detectó un segundo objeto para medir.")

    else:
        print("El radio detectado es muy pequeño. Asegúrate de que la moneda esté bien visible.")
else:
    print("No se detectaron contornos.")

# Paso 7: Mostrar resultados
cv2.imshow("Resultado", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
