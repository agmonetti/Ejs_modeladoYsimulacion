# Importaciones necesarias
import numpy as np
from tabulate import tabulate

def biseccion(f, a, b, iteraciones=100, tolerancia=1e-6, precision=5):

    # Verificación inicial
    if f(a) * f(b) >= 0:
        raise ValueError("La función debe tener signos opuestos...")

    results = []

    for i in range(iteraciones):
        c = (a + b) / 2.0
        fc = f(c)

        results.append([i+1, round(a, precision), round(b, precision), round(c, precision), round(fc, precision)])
        print(tabulate(results, headers=["i", "a", "b", "c", "f(c)"]))

        # Condición de parada
        if abs(fc) < tolerancia or (b - a) / 2.0 < tolerancia:
            return c

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    raise ValueError("El método no convergió...")
# Definir la función
def f(x):
    return 0.4 * np.exp(x**2) - 5*x

# Intervalo inicial
a = 1
b = 2

# Encontrar y mostrar la raíz
raiz = biseccion(f, a, b)
print(f"La raíz encontrada es: {raiz}")

