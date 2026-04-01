import numpy as np


def funcion(x):
    return 6 + 3 * np.cos(x)

# calculamos la segunda derivada numéricamente para calcular el error de truncamiento
def segunda_derivada_numerica(f, x, dx=1e-5):
    """Aproxima la segunda derivada usando diferencias centrales finitas."""
    return (f(x + dx) - 2 * f(x) + f(x - dx)) / (dx**2)

# Límites de integración y subintervalos
a = 0
b = np.pi/2
n = 4 # numero de divisiones (entre mas divisiones, mas preciso, pero mas calculos)

# Paso
h = (b - a) / n

# --- CÁLCULO DE LA INTEGRAL ---
x = np.linspace(a, b, n + 1)
y = funcion(x)

# Regla del trapecio compuesta
integral = (h / 2) * (y[0] + 2 * np.sum(y[1:n]) + y[-1])


# --- CÁLCULO DEL ERROR MÁXIMO AUTOMATIZADO ---
# Creamos el vector de "escaneo"
x_fino = np.linspace(a, b, 1000) 

# Le pasamos tu 'funcion' al calculador numérico. 
# Como x_fino es un vector de numpy, calcula los 1000 puntos al mismo tiempo.
arreglo_segundas_derivadas = segunda_derivada_numerica(funcion, x_fino)

# Buscamos el valor más alto en valor absoluto
max_segunda_derivada = np.max(np.abs(arreglo_segundas_derivadas))

# fórmula teórica del error
cota_error = ((b - a)**3 / (12 * n**2)) * max_segunda_derivada


# --- IMPRESIÓN DE RESULTADOS ---
print(f"Integral aproximada (Trapecio Compuesta): {integral:.8f}")
print(f"Cota de Error de Truncamiento máximo  : ±{cota_error:.8f}")