import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,recall_score
import matplotlib.pyplot as plt

x=np.array([[1],[2],[3]]) #Matriz
y=np.array([2,3.5,5.5]) #vector
#declaracion
valor=np.array([[2]])
valor2=np.array([[3.75]])
modelo=linear_model.LinearRegression()
modelo.fit(x,y)
print(modelo.predict(valor))
print(modelo.predict(valor2))
print((mean_squared_error(y_true=y,y_pred=modelo.predict(x))))
#print(recall_score(y, modelo.predict(x)))
#graficar
color="red"
plt.scatter(x,y, c=color)
plt.plot(x,modelo.predict(x))
plt.scatter(x,y)
plt.show()
