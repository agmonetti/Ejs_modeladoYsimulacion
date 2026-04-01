import numpy as np

# Definimos la función a integrar
def funcion(x):
    return np.exp(x**2) 

# Límites de integración
a = 0 # Límite inferior
b = 1 # Límite superior

# Número de subintervalos
n = 4 

# Paso
h = (b - a) / n

# Puntos medios
x_medio = np.linspace(a + h/2, b - h/2, n)

# Aplicamos la regla del rectángulo medio compuesta
integral = h * np.sum(funcion(x_medio))

print(f"Integral aproximada con la regla del rectángulo medio compuesta: {integral:.8f}")
# Si quiero mas o menos decimales, cambio el formato en el print, por ejemplo: .4f para 4 decimales.