import numpy as np

# Definimos la función a integrar
def funcion(x):
    return 6 + 3*np.cos(x)

# Límites de integración
a = 0 # Límite inferior
b = np.pi/2 # Límite superior

# Evaluamos la función en los extremos
fa = funcion(a)
fb = funcion(b)

# Aplicamos la regla del trapecio simple
integral = (b - a) / 2 * (fa + fb)

print(f"Integral aproximada con la regla del trapecio simple: {integral:.8f}")
# Si quiero mas o menos decimales, cambio el formato en el print, por ejemplo: .4f para 4 decimales.