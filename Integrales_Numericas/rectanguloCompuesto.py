import numpy as np
from tabulate import tabulate

# 1. Definimos la función a integrar (Ejemplo de la pizarra)
def funcion(x):
    return np.exp(x**2)

# 2. Cálculo automático del error (Segunda derivada)
def segunda_derivada_numerica(f, x, dx=1e-5):
    """Aproxima la segunda derivada usando diferencias centrales finitas."""
    return (f(x + dx) - 2 * f(x) + f(x - dx)) / (dx**2)

def rectangulo_medio_compuesto_pizarra(f, a, b, n, precision=6):
    print("\n" + "="*60)
    print(" MÉTODO: RECTÁNGULO MEDIO COMPUESTO ")
    print("="*60)
    
    h = (b - a) / n
    print(f"Parámetros: n = {n}  ->  h = {h}")
    
    # Vectores: Nodos normales y Nodos medios
    x = np.linspace(a, b, n + 1)
    x_medio = np.linspace(a + h/2, b - h/2, n)
    y_medio = f(x_medio)
    
    # --- CONSTRUCCIÓN DE LA TABLA ---
    tabla_pizarra = []
    # Fila 0 con guiones para los puntos medios (alineado a la pizarra)
    tabla_pizarra.append([0, round(x[0], precision), "-", "-"])
    
    for i in range(1, n + 1):
        tabla_pizarra.append([
            i, 
            round(x[i], precision), 
            round(x_medio[i-1], precision), 
            round(y_medio[i-1], precision)
        ])
        
    print("\nTABLA DE VALORES:")
    print(tabulate(tabla_pizarra, headers=["n", "x_n", "x_medio_n", "f(x_medio_n)"], tablefmt="grid",disable_numparse=True))
    
    # --- DESARROLLO DE LA FÓRMULA ---
    str_valores = " + ".join([f"{val:.{precision}f}" for val in y_medio])
    desarrollo = f"A ~= {h} [ {str_valores} ]"
    
    print("\nDESARROLLO DE LA FÓRMULA:")
    print(desarrollo)
    
    # --- CÁLCULO NUMÉRICO Y ERROR ---
    integral = h * np.sum(y_medio)
    
    x_fino = np.linspace(a, b, 1000) 
    max_segunda = np.max(np.abs(segunda_derivada_numerica(f, x_fino)))
    cota_error = ((b - a)**3 / (24 * n**2)) * max_segunda
    
    print(f"\nRESULTADOS FINALES:")
    print(f"A ~= {integral:.8f}") 
    print(f"E_t <= ±{cota_error:.8f}")
    print("="*60 + "\n")
    
    return integral, cota_error

# --- BLOQUE PRINCIPAL ---
if __name__ == "__main__":
    a = 0
    b = 1
    n = 4 # Número de subintervalos

    decimales = 8
    
    resultado, error = rectangulo_medio_compuesto_pizarra(funcion, a, b, n, precision=decimales)