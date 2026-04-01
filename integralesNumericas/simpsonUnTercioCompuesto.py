import numpy as np

# 1. Definimos la función a integrar
def funcion(x):
    return 6 + 3 * np.cos(x)

# 2. El "Auto-Calculador" de la CUARTA derivada numérica
def cuarta_derivada_numerica(f, x, dx=1e-3):
    """Aproxima la cuarta derivada usando diferencias centrales."""
    # Nota técnica: Usamos dx=1e-3 en lugar de 1e-5 porque al elevarse a la cuarta potencia (dx**4), 
    # un número muy chico causaría un desbordamiento a cero en la memoria (underflow).
    return (f(x + 2*dx) - 4*f(x + dx) + 6*f(x) - 4*f(x - dx) + f(x - 2*dx)) / (dx**4)

# Límites de integración
a = 0
b = np.pi/2

# Número de subintervalos (DEBE SER PAR)
n = 4

# --- VALIDACIÓN TEÓRICA ---
if n % 2 != 0:
    raise ValueError("¡Error! La regla de Simpson 1/3 exige que el número de subintervalos (n) sea PAR.")

# Paso
h = (b - a) / n

# --- CÁLCULO DE LA INTEGRAL ---
x = np.linspace(a, b, n + 1)
y = funcion(x)

# Aplicamos la regla de Simpson compuesta con slicing de NumPy
S = y[0] + y[-1] + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2:n-1:2])
integral = (h / 3) * S


# --- CÁLCULO DEL ERROR MÁXIMO AUTOMATIZADO ---
# Creamos el vector de "escaneo"
x_fino = np.linspace(a, b, 1000) 

# Auto-calculamos la 4ta derivada para toda la curva
arreglo_cuartas_derivadas = cuarta_derivada_numerica(funcion, x_fino)

# Buscamos el valor más alto en valor absoluto (el peor escenario)
max_cuarta_derivada = np.max(np.abs(arreglo_cuartas_derivadas))

# Aplicamos la fórmula teórica del error para Simpson 1/3
cota_error = ((b - a)**5 / (180 * n**4)) * max_cuarta_derivada


# --- IMPRESIÓN DE RESULTADOS ---
print(f"Integral aproximada (Simpson 1/3 Compuesta) : {integral:.8f}")
print(f"Cota de Error de Truncamiento máximo        : ±{cota_error:.8e}")