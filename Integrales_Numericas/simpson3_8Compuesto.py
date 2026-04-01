import numpy as np

# 1. Definimos la función a integrar
def funcion(x):
    return np.sin(x)

# 2. Helper para el cálculo automático del error (Cuarta derivada)
def cuarta_derivada_numerica(f, x, dx=1e-3):
    """Aproxima la cuarta derivada usando diferencias centrales finitas."""
    return (f(x + 2*dx) - 4*f(x + dx) + 6*f(x) - 4*f(x - dx) + f(x - 2*dx)) / (dx**4)

def simpson_38_compuesto(f, a, b, n, precision=6):
    print("\n" + "="*60)
    print(" MÉTODO: SIMPSON 3/8 COMPUESTO ")
    print("="*60)
    
    # Validación teórica estricta
    if n % 3 != 0:
        raise ValueError("¡Error! La regla de Simpson 3/8 exige que el número de subintervalos (n) sea MÚLTIPLO DE 3.")

    h = (b - a) / n
    print(f"Parámetros: n = {n}  ->  h = {h}")
    
    # Generamos los vectores
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    # --- 1. AGRUPACIÓN Y DESARROLLO DE LA FÓRMULA ---
    # Extraemos los grupos usando slicing
    grupo_1 = y[1:n:3]
    grupo_2 = y[2:n:3]
    grupo_3 = y[3:n-1:3]
    
    # Convertimos a texto para imprimir el desarrollo
    str_1 = " + ".join([f"{val:.{precision}f}" for val in grupo_1])
    str_2 = " + ".join([f"{val:.{precision}f}" for val in grupo_2])
    str_3 = " + ".join([f"{val:.{precision}f}" for val in grupo_3])
    
    fraccion_h = f"3*{h}/8" if isinstance(h, float) else f"3({b}-{a})/{8*n}"
    
    # Manejamos el caso donde el grupo 3 (los múltiplos de 3) esté vacío (ej. si n=3)
    termino_mult3 = f" + 2({str_3})" if str_3 else ""
    desarrollo = f"I ~= {fraccion_h} [ {y[0]:.{precision}f} + 3({str_1}) + 3({str_2}){termino_mult3} + {y[-1]:.{precision}f} ]"
    
    print("\nDESARROLLO DE LA FÓRMULA:")
    print(desarrollo)
    
    # --- 2. CÁLCULO NUMÉRICO DE LA INTEGRAL ---
    S = y[0] + y[-1] + 3 * np.sum(grupo_1) + 3 * np.sum(grupo_2) + 2 * np.sum(grupo_3)
    integral = (3 * h / 8) * S
    
    # --- 3. CÁLCULO DEL ERROR DE TRUNCAMIENTO MÁXIMO ---
    x_fino = np.linspace(a, b, 1000) 
    arreglo_cuartas_derivadas = cuarta_derivada_numerica(f, x_fino)
    max_cuarta = np.max(np.abs(arreglo_cuartas_derivadas))
    
    # Fórmula correcta del error compuesto para Simpson 3/8
    cota_error = ((b - a)**5 / (80 * n**4)) * max_cuarta
    
    print(f"\nRESULTADOS FINALES:")
    print(f"I ~= {integral:.{precision}f}")
    print(f"E_t <= ±{cota_error:.8e}")
    print("="*60 + "\n")
    
    return integral, cota_error


# --- BLOQUE PRINCIPAL (UNIFICADO) ---
if __name__ == "__main__":
    # Parámetros centralizados
    a = 0
    b = np.pi
    n = 9 # Debe ser múltiplo de 3
    decimales = 6
    
    resultado, error = simpson_38_compuesto(funcion, a, b, n, precision=decimales)