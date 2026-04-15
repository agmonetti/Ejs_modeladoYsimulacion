import numpy as np
from tabulate import tabulate

# 1. Definimos la función a integrar (Ejemplo de la pizarra)
def funcion(x):
    return (x**x)

# 2. Helper para el cálculo automático del error (Cuarta derivada)
def cuarta_derivada_numerica(f, x, dx=1e-3):
    """Aproxima la cuarta derivada usando diferencias centrales."""
    return (f(x + 2*dx) - 4*f(x + dx) + 6*f(x) - 4*f(x - dx) + f(x - 2*dx)) / (dx**4)

def simpson_13_compuesto_pizarra(f, a, b, n, precision=6):
    print("\n" + "="*55)
    print(" MÉTODO: SIMPSON 1/3 COMPUESTO (INFORME COMPLETO) ")
    print("="*55)
    
    # Validación teórica inquebrantable
    if n % 2 != 0:
        raise ValueError("¡Error! La regla de Simpson 1/3 exige que el número de subintervalos (n) sea PAR.")

    h = (b - a) / n
    print(f"Parámetros: n = {n}  ->  h = {h}")
    
    # Generamos los vectores
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    # --- 1. CONSTRUCCIÓN DE LA TABLA EXACTA ---
    tabla_pizarra = []
    for i in range(n + 1):
        tabla_pizarra.append([i, round(x[i], precision), round(y[i], precision)])
        
    print("\nTABLA DE VALORES:")
    print(tabulate(tabla_pizarra, headers=["n", "x_n", "f(x_n)"], tablefmt="grid",disable_numparse=True))
    
    # --- 2. DESARROLLO ESCRITO DE LA FÓRMULA ---
    # Extraemos pares e impares usando slicing
    impares = y[1:n:2]
    pares = y[2:n-1:2]
    
    # Convertimos a texto con la precisión deseada
    str_impares = " + ".join([f"{val:.{precision}f}" for val in impares])
    str_pares = " + ".join([f"{val:.{precision}f}" for val in pares])
    
    fraccion_h = f"{h}/3" if isinstance(h, float) else f"({b}-{a})/{3*n}"
    
    # Construimos el desarrollo condicionalmente (por si n=2 y no hay términos pares)
    termino_pares = f" + 2({str_pares})" if str_pares else ""
    desarrollo = f"I ~= {fraccion_h} [ {y[0]:.{precision}f} + 4({str_impares}){termino_pares} + {y[-1]:.{precision}f} ]"
    
    print("\nDESARROLLO DE LA FÓRMULA:")
    print(desarrollo)
    
    # --- 3. CÁLCULO NUMÉRICO DE LA INTEGRAL ---
    S = y[0] + y[-1] + 4 * np.sum(impares) + 2 * np.sum(pares)
    integral = (h / 3) * S
    
    # --- 4. CÁLCULO DEL ERROR DE TRUNCAMIENTO MÁXIMO ---
    x_fino = np.linspace(a, b, 1000) 
    arreglo_cuartas_derivadas = cuarta_derivada_numerica(f, x_fino)
    max_cuarta = np.max(np.abs(arreglo_cuartas_derivadas))
    
    # Fórmula del error: ((b-a)^5 / 180n^4) * max|f''''(x)|
    cota_error = ((b - a)**5 / (180 * n**4)) * max_cuarta
    
    print(f"\nRESULTADOS FINALES:")
    print(f"I ~= {integral:.{precision}f}")
    # Usamos notación científica explícita para el error si es muy pequeño, o más decimales
    print(f"E_t <= ±{cota_error:.8f}")
    print("="*55 + "\n")
    
    return integral, cota_error

# --- BLOQUE PRINCIPAL ---
if __name__ == "__main__":
    # Variables de control del ejercicio de la pizarra
    a = 0
    b = 1
    n = 4 # Número de subintervalos
    
    decimales = 8
    
    resultado, error = simpson_13_compuesto_pizarra(funcion, a, b, n, precision=decimales)