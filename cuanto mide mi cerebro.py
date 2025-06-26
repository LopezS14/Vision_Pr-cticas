import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Cargar datos
data = pd.read_csv('brain_body (1).txt', delim_whitespace=True)

data['Height'] = 158  # Aquí deberías colocar tus datos de estatura

# Variables independientes (Body y Height) y dependiente (Brain)
X = data[['Body', 'Height']]  # Ahora estamos considerando Body y Height como variables independientes
y = data['Brain']

# Ajustar modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Mostrar resultados
print("Coeficientes:", modelo.coef_)
print("Intercepto:", modelo.intercept_)

# Graficar
plt.scatter(data['Body'], data['Brain'], color='blue', label='Datos')
plt.plot(data['Body'], modelo.predict(X), color='red', label='Regresión lineal inversa')
plt.xlabel('Masa corporal (Body)')
plt.ylabel('Masa cerebral (Brain)')
plt.legend()
plt.title('Regresión lineal inversa entre masa corporal y cerebral con estatura')
plt.show()

# Calcular la masa cerebral para un valor específico de masa corporal y estatura
# Por ejemplo, si quieres saber el cerebro de una persona con 70 kg de masa corporal y 170 cm de estatura
masa_corporal = 70
estatura = 170
masa_cerebral_predicha = modelo.predict([[masa_corporal, estatura]])
print("Masa cerebral predicha:", masa_cerebral_predicha)
