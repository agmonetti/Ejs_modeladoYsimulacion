import numpy as np
from tabulate import tabulate

# 1. Definimos la función a integrar
def funcion(x):
    return np.exp(x**4)

# 2. Helper para el cálculo automático del error
def segunda_derivada_numerica(f, x, dx=1e-5):
    """Aproxima la segunda derivada usando diferencias centrales finitas."""
    return (f(x + dx) - 2 * f(x) + f(x - dx)) / (dx**2)

def trapecio_compuesto_pizarra(f, a, b, n, precision=8):
    print("\n" + "="*55)
    print(" MÉTODO: TRAPECIO COMPUESTO (INFORME COMPLETO) ")
    print("="*55)
    
    h = (b - a) / n
    print(f"Parámetros: n = {n}  ->  h = {h}")
    
    # Generamos los vectores
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    # --- 1. CONSTRUCCIÓN DE LA TABLA EXACTA ---
    tabla_pizarra = []
    for i in range(n + 1):
        tabla_pizarra.append([i, f"{x[i]:.{precision}f}", f"{y[i]:.{precision}f}"])
        
    print("\nTABLA DE VALORES:")
    print(tabulate(tabla_pizarra, headers=["n", "x_n", "f(x_n)"], tablefmt="grid", disable_numparse=True))
    
    # --- 2. DESARROLLO ESCRITO DE LA FÓRMULA ---
    interiores = y[1:n]
    str_interiores = " + ".join([f"{val:.{precision}f}" for val in interiores])
    
    fraccion_h = f"{h}/2" if isinstance(h, float) else f"({b}-{a})/{2*n}"
    desarrollo = f"I ~= {fraccion_h} [ {y[0]:.{precision}f} + 2({str_interiores}) + {y[-1]:.{precision}f} ]"
    
    print("\nDESARROLLO DE LA FÓRMULA:")
    print(desarrollo)
    
    # --- 3. CÁLCULO NUMÉRICO DE LA INTEGRAL ---
    integral = (h / 2) * (y[0] + 2 * np.sum(interiores) + y[-1])
    
    # --- 4. CÁLCULO DEL ERROR DE TRUNCAMIENTO MÁXIMO ---
    x_fino = np.linspace(a, b, 1000) 
    arreglo_segundas_derivadas = segunda_derivada_numerica(f, x_fino)
    max_segunda_derivada = np.max(np.abs(arreglo_segundas_derivadas))
    
    # Fórmula del error: ((b-a)^3 / 12n^2) * max|f''(x)|
    cota_error = ((b - a)**3 / (12 * n**2)) * max_segunda_derivada
    
    print(f"\nRESULTADOS FINALES:")
    print(f"I ~= {integral:.{precision}f}")
    print(f"E_t <= ±{cota_error:.{precision}f}")
    print("="*55 + "\n")
    
    return integral, cota_error

# --- BLOQUE PRINCIPAL ---
if __name__ == "__main__":
    # Variables de control
    a = -1
    b = 1
    n = 5 # Número de subintervalos
    
    # Control centralizado de la cantidad de decimales
    decimales = 12
    
    resultado, error = trapecio_compuesto_pizarra(funcion, a, b, n, precision=decimales)