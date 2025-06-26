import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Cargar datos
data = pd.read_csv('brain_body (1).txt', delim_whitespace=True)
X = data[['Brain']]  # Asegúrate que sea un DataFrame (dobles corchetes)
y = data['Body']

# Ajustar modelo
modelo=LinearRegression()
modelo.fit(X, y)
# Mostrar resultados
print("Coeficiente:", modelo.coef_[0])
print("Intercepto:", modelo.intercept_)

# Graficar
plt.scatter(X, y, color='blue', label='Datos')
plt.plot(X, modelo.predict(X), color='red', label='Regresión lineal')
plt.xlabel('Masa cerebral (Brain)')
plt.ylabel('Masa corporal (Body)')
plt.legend()
plt.title('Regresión lineal entre masa cerebral y corporal')
plt.show()
