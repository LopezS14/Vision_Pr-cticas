import numpy as np
import matplotlib.pyplot as plt
#Librerias de la red neuronal
from keras import backend
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn import linear_model

x = np.array([[1], [2], [3]])  # matriz
y = np.array([2, 3.5, 5.5])    # vector
#predicir valor cualquiera
valor = np.array([[4], [5], [6]])

modelo=linear_model.LinearRegression()
modelo.fit(x,y)
# limpiar sesión
backend.clear_session()

# crear el modelo
modelo = Sequential()
modelo.add(Dense(1, use_bias=True, activation='linear', input_shape=(1,)))

# optimizador (no es necesario decay)
adam = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

# compilar modelo
modelo.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

# entrenar
modelo.fit(x, y, epochs=200, verbose=0, validation_data=(x, y))
#predecir valores
# resumen y resultados
modelo.summary()
score = modelo.evaluate(x, y)
print("Pérdida y métrica:", score)
print("Pesos:", modelo.get_weights()[0])
print("Sesgo:", modelo.get_weights()[1])
print("Valor predicho",modelo.predict(valor))

