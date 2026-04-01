import numpy as np

# Definimos la función a integrar
def funcion(x):
    return np.sin(x) # Puedes cambiarla por cualquier otra función

# Límites de integración
a = 0 # Límite inferior
b = np.pi # Límite superior

# Punto medio
m = (a + b) / 2

# Evaluamos la función en los tres puntos
fa = funcion(a)
fm = funcion(m)
fb = funcion(b)

# Aplicamos la regla de Simpson simple
integral = (b - a) / 6 * (fa + 4 * fm + fb)

print(f"Integral aproximada con Simpson simple: {integral:.4f}")
