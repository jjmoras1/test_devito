```python
import cv2 # librería para digitalizar imagenes
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd


```

# Importación Barra de color


```python
# Cargar la imagen
img = cv2.imread('barraDeColor.jpg')



```


```python
# Convertir la imagen de BGR (formato de OpenCV) a RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```


```python
# Mostrar la imagen original para ver su formato
plt.imshow(img_rgb)
plt.title('Imagen Original')
plt.show()
```


    
![png](BarraDeColor_files/BarraDeColor_4_0.png)
    



```python
plt.imshow(img,aspect='auto',interpolation='nearest')
```




    <matplotlib.image.AxesImage at 0x7f42a2d824a0>




    
![png](BarraDeColor_files/BarraDeColor_5_1.png)
    


# Crear Modelo Prediicion vel color


```python

```


```python

```


```python
# Normalizar los valores de la imagen entre 0 y 1
img_normalizada = img_rgb / 255.0  # La imagen se normaliza para estar en el rango [0, 1]
```


```python

```

Normalización Barra


```python
vel_min=5710*0.3048/1000 #vel Km/s
vel_min
```




    1.7404080000000002




```python
vel_max=16000*0.3048/1000 #vel Km/s
vel_max
```




    4.8768




```python
velocidad_matriz = vel_min + (img_normalizada * (vel_max - vel_min))
```


```python
# Mostrar la matriz de velocidades (un solo canal, por ejemplo, el canal R de RGB)
plt.imshow(velocidad_matriz[:, :, 0], cmap='inferno')  # Usamos un canal para mostrar como escala
plt.colorbar(label='Velocidad (Km/s)')
plt.title('Campo de Velocidades a partir de la Imagen')
plt.show()
```


    
![png](BarraDeColor_files/BarraDeColor_15_0.png)
    



```python
matriz_velocidad_final = velocidad_matriz[:, :, 0]  # Solo tomamos el canal R de RGB
```


```python
matriz_velocidad_final.min()
```




    1.7404080000000002




```python

matriz_velocidad_final.max()
 
```




    4.8768




```python
plt.imshow(matriz_velocidad_final,aspect='auto',interpolation='nearest')
```




    <matplotlib.image.AxesImage at 0x7f42a2d2efe0>




    
![png](BarraDeColor_files/BarraDeColor_19_1.png)
    


# Importación campo de vel


```python
CampoVel = cv2.imread('campo_vel.jpg')
```


```python
CampoVel.shape
```




    (695, 1109, 3)




```python
plt.imshow(CampoVel, aspect='auto', cmap='viridis', interpolation='nearest')
```




    <matplotlib.image.AxesImage at 0x7f42a2bb7400>




    
![png](BarraDeColor_files/BarraDeColor_23_1.png)
    



```python

# Cargar la imagen


# Convertir la imagen de BGR (formato de OpenCV) a RGB
CampoVel_rgb = cv2.cvtColor(CampoVel, cv2.COLOR_BGR2RGB)
# Mostrar la imagen original para ver su formato
plt.imshow(CampoVel_rgb)
plt.title('Imagen Original')
plt.show()


```


    
![png](BarraDeColor_files/BarraDeColor_24_0.png)
    



```python
# Normalizar los valores de la imagen entre 0 y 1
CampoVel_normalizada = CampoVel_rgb / 255.0  # La imagen se normaliza para estar en el rango [0, 1]
```


```python
print(CampoVel_rgb.shape)
print(CampoVel.shape)
```

    (695, 1109, 3)
    (695, 1109, 3)



```python
# Mapear los valores normalizados a la escala de velocidad
# Usamos una fórmula de escalado lineal
velocidad_matriz = vel_max + (vel_min-vel_max)*CampoVel_normalizada
```


```python







# Mostrar la matriz de velocidades (un solo canal, por ejemplo, el canal R de RGB)
plt.imshow(velocidad_matriz[:, :, 0], cmap='seismic')  # Usamos un canal para mostrar como escala
plt.colorbar(label='Velocidad (m/s)')
plt.title('Campo de Velocidades a partir de la Imagen')
plt.show()

# Ahora tienes la matriz de velocidad_matriz que contiene los valores de velocidad
# Puedes extraer un canal si solo necesitas uno, por ejemplo, el canal R:
matriz_velocidad_final = velocidad_matriz[:, :, 0]  # Solo tomamos el canal R de RGB
```


    
![png](BarraDeColor_files/BarraDeColor_28_0.png)
    



```python

```

# Atar modelo de velocidad a barra de colores
Se pudo observar que la barra de colores está discreizada en 20 pedazos cada uno con un vector rgb [R,G,B]. Las normalizaciones anteriores asumian un gradiente lineal ente el color más claro con el más oscuro ente el rango de velocidades. Como vamos a ver esto no es cierto ya que el comportamiento de los valores RGB no es lineal. Para encontrar un mejor modelo vamos a hacer una regresión para poder encontrar un mejor modelo entre colores y velocidades


```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
```

En este dataframe se muestran los valores RGB vs el valor correspondiente a cada velocidad


```python
df=pd.read_csv('Barra_de_color.csv',sep=";")


# Supongamos que df es tu DataFrame y 'v' es la columna objetivo
df['vel'] = df['vel'].str.replace(',', '.', regex=False).astype(float)

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vel</th>
      <th>R</th>
      <th>G</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.740408</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.905481</td>
      <td>246</td>
      <td>246</td>
      <td>218</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.070555</td>
      <td>239</td>
      <td>237</td>
      <td>184</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.235628</td>
      <td>233</td>
      <td>227</td>
      <td>156</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.400701</td>
      <td>229</td>
      <td>219</td>
      <td>126</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.565774</td>
      <td>224</td>
      <td>211</td>
      <td>103</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2.730848</td>
      <td>220</td>
      <td>202</td>
      <td>88</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2.895921</td>
      <td>213</td>
      <td>183</td>
      <td>80</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3.060994</td>
      <td>203</td>
      <td>150</td>
      <td>71</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3.226067</td>
      <td>197</td>
      <td>121</td>
      <td>64</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3.391141</td>
      <td>197</td>
      <td>115</td>
      <td>61</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3.556214</td>
      <td>195</td>
      <td>127</td>
      <td>62</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3.721287</td>
      <td>179</td>
      <td>136</td>
      <td>60</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3.886360</td>
      <td>157</td>
      <td>111</td>
      <td>51</td>
    </tr>
    <tr>
      <th>14</th>
      <td>4.051434</td>
      <td>133</td>
      <td>82</td>
      <td>40</td>
    </tr>
    <tr>
      <th>15</th>
      <td>4.216507</td>
      <td>114</td>
      <td>57</td>
      <td>32</td>
    </tr>
    <tr>
      <th>16</th>
      <td>4.381580</td>
      <td>88</td>
      <td>41</td>
      <td>25</td>
    </tr>
    <tr>
      <th>17</th>
      <td>4.546653</td>
      <td>56</td>
      <td>24</td>
      <td>16</td>
    </tr>
    <tr>
      <th>18</th>
      <td>4.711727</td>
      <td>25</td>
      <td>7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19</th>
      <td>4.876800</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X=df[['R','G','B']]
y=df['vel']
```


```python
import matplotlib.pyplot as plt

# Crear figuras y ejes
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Gráfico r vs y
axs[0].scatter(df['R'], y, color='red')
axs[0].set_xlabel('R')
axs[0].set_ylabel('V')
axs[0].set_title('R vs V')

# Gráfico g vs y
axs[1].scatter(df['G'], y, color='green')
axs[1].set_xlabel('G')
axs[1].set_ylabel('V')
axs[1].set_title('G vs V')

# Gráfico b vs y
axs[2].scatter(df['B'], y, color='blue')
axs[2].set_xlabel('B')
axs[2].set_ylabel('V')
axs[2].set_title('B vs V')

plt.tight_layout()
plt.show()


```


    
![png](BarraDeColor_files/BarraDeColor_35_0.png)
    


Claramente se puede observar que el comportamiento RGV con respecto a la velocidad no es lineal. Lo que ratifica que una interpolacion lineal de colores estaba equivocada


```python
X=df[['R','G','B']]
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```


```python
# Modelo de Regresión Lineal
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```


```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Calcular MSE
mse = mean_squared_error(y_test, y_pred)

# Calcular RMSE
rmse = np.sqrt(mse)

# Calcular MAE
mae = mean_absolute_error(y_test, y_pred)

# Calcular R^2
r2 = r2_score(y_test, y_pred)

# Imprimir los resultados
print("Métricas de rendimiento:")
print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
print(f"Error Absoluto Medio (MAE): {mae:.4f}")
print(f"R^2: {r2:.4f}")

```

    Métricas de rendimiento:
    Error Cuadrático Medio (MSE): 0.0271
    Raíz del Error Cuadrático Medio (RMSE): 0.1646
    Error Absoluto Medio (MAE): 0.1369
    R^2: 0.9763


Interpretación de las Métricas
Error Cuadrático Medio (MSE):

Valor: 0.0271
Indica que, en promedio, el cuadrado de los errores de predicción es 0.0271. Un valor bajo sugiere que el modelo hace predicciones bastante cercanas a los valores reales.
Raíz del Error Cuadrático Medio (RMSE):

Valor: 0.1646
Este valor indica que la desviación estándar de los errores de predicción es aproximadamente 0.1646. Al estar en las mismas unidades que tu variable objetivo v, proporciona una forma intuitiva de entender el tamaño del error.
Error Absoluto Medio (MAE):

Valor: 0.1369
Este valor representa el error promedio en términos absolutos. Significa que, en promedio, tus predicciones se desvían de los valores reales en aproximadamente 0.1369.
R^2 (Coeficiente de Determinación):

Valor: 0.9763
Este valor indica que el 97.63% de la varianza en la variable v puede ser explicada por las variables independientes r, g, y b. Un R^2 cercano a 1 indica un modelo que se ajusta muy bien a los datos.


```python
import matplotlib.pyplot as plt

# Graficar los valores reales vs predicciones
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)  # Línea de referencia
plt.xlabel('Valores Reales (y_test)')
plt.ylabel('Predicciones (y_pred)')
plt.title('Valores Reales vs Predicciones')

plt.show()

```


    
![png](BarraDeColor_files/BarraDeColor_42_0.png)
    


En este grafico podemos ver que los valores predichos con respecto a los reales se acercan bastante

red nero


```python
CampoVel_rgb[0][0]
```




    array([243, 249, 247], dtype=uint8)




```python
X.shape
```




    (20, 3)




```python
X.iloc[0].shape
```




    (3,)




```python
df_test=pd.DataFrame(X.iloc[0])
df_test.shape
```




    (3, 1)




```python
a=CampoVel_rgb[0][0].reshape((1, 3))
a.shape
```




    (1, 3)




```python

def vel_pred(matriz_rgb):
    # Convertir la matriz ixjx3 en un DataFrame de una sola vez
    matriz_reshaped = matriz_rgb.reshape(-1, 3)  # Convierte la matriz en 4x3 (sin bucles)
    # Suponiendo que b es la matriz de RGB y reg es el modelo de regresión lineal ya entrenado
    df_b = pd.DataFrame(matriz_reshaped, columns=['R', 'G', 'B'])
    # Realizar la predicción de todo el DataFrame en un solo paso
    predicciones = reg.predict(df_b)
    # Convertir las predicciones a una matriz 2x2
    c = predicciones.reshape(matriz_rgb.shape[0], matriz_rgb.shape[1])

    return c
    











```


```python
v_final=vel_pred(CampoVel)
```


```python
v_final.shape
```




    (695, 1109)




```python
v_final
```




    array([[1.56901352, 1.55539341, 1.54125388, ..., 1.47414717, 1.47414717,
            1.47414717],
           [1.55487399, 1.54125388, 1.54125388, ..., 1.47821252, 1.47821252,
            1.47821252],
           [1.54683249, 1.53321238, 1.51959227, ..., 1.47821252, 1.49183263,
            1.47821252],
           ...,
           [4.92003521, 4.92003521, 4.92003521, ..., 4.92003521, 4.92003521,
            4.92003521],
           [4.92003521, 4.92003521, 4.92003521, ..., 4.92003521, 4.92003521,
            4.92003521],
           [4.92003521, 4.92003521, 4.92003521, ..., 4.92003521, 4.92003521,
            4.92003521]])




```python


# Supongamos que 'v_final' es tu matriz que quieres mostrar
plt.figure(figsize=(8, 6))

# Crear la imagen con el mapa de color 'inferno'
plt.imshow(v_final, aspect='auto', cmap='jet', interpolation='nearest')

# Añadir una barra de color
cbar = plt.colorbar()
cbar.set_label('V (Km/s)')

# Añadir título
plt.title('Visualización de campo de velocidades')

# Mostrar el gráfico
plt.show()

```


    
![png](BarraDeColor_files/BarraDeColor_54_0.png)
    

