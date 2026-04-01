import numpy as np
from tabulate import tabulate

'''
Archivo Consolidador: Métodos de Integración Numérica (Fórmulas de Newton-Cotes)
    - Regla del Rectángulo Medio Compuesto
    - Regla del Trapecio Compuesto
    - Regla de Simpson 1/3 Compuesto

    Resultado por cada método:
    * Parámetros utilizados (a, b, n, h)
    * Tabla de iteraciones/evaluaciones (Formato Pizarra UADE)
    * Desarrollo algebraico de la sumatoria
    * Cálculo de la integral aproximada
    * Cálculo de la cota máxima de Error de Truncamiento (E_t)
'''

# -----------------------------------
#  FUNCION A INTEGRAR
# ------------------------------------
def funcion(x):
    return (x**2) * np.exp(x) 


# -----------------------------------
#  DERIVADAS NUMÉRICAS (Para el Error)
# ------------------------------------
def segunda_derivada_numerica(f, x, dx=1e-5):
    """Aproxima la 2da derivada (Usada en Rectángulo y Trapecio)."""
    return (f(x + dx) - 2 * f(x) + f(x - dx)) / (dx**2)

def cuarta_derivada_numerica(f, x, dx=1e-3):
    """Aproxima la 4ta derivada (Usada en Simpson 1/3)."""
    return (f(x + 2*dx) - 4*f(x + dx) + 6*f(x) - 4*f(x - dx) + f(x - 2*dx)) / (dx**4)


# -----------------------------------
#  MÉTODOS NUMÉRICOS
# ------------------------------------

def rectangulo_medio_compuesto(f, a, b, n, precision=6):
    print("\n" + "="*60)
    print(" 1. MÉTODO: RECTÁNGULO MEDIO COMPUESTO ")
    print("="*60)
    
    h = (b - a) / n
    print(f"Parámetros: n = {n}  ->  h = {h}")
    
    x = np.linspace(a, b, n + 1)
    x_medio = np.linspace(a + h/2, b - h/2, n)
    y_medio = f(x_medio)
    
    # Tabla
    tabla = [[0, round(x[0], precision), "-", "-"]]
    for i in range(1, n + 1):
        tabla.append([i, round(x[i], precision), round(x_medio[i-1], precision), round(y_medio[i-1], precision)])
        
    print("\nTABLA DE VALORES:")
    print(tabulate(tabla, headers=["n", "x_n", "x_medio_n", "f(x_medio_n)"], tablefmt="grid"))
    
    # Desarrollo
    str_valores = " + ".join([f"{val:.{precision}f}" for val in y_medio])
    print(f"\nDESARROLLO: A ~= {h} [ {str_valores} ]")
    
    # Cálculos
    integral = h * np.sum(y_medio)
    max_segunda = np.max(np.abs(segunda_derivada_numerica(f, np.linspace(a, b, 1000))))
    cota_error = ((b - a)**3 / (24 * n**2)) * max_segunda
    
    print(f"\nRESULTADOS: I ~= {integral:.8f}  |  E_t <= ±{cota_error:.8f}")
    return integral, cota_error


def trapecio_compuesto(f, a, b, n, precision=6):
    print("\n" + "="*60)
    print(" 2. MÉTODO: TRAPECIO COMPUESTO ")
    print("="*60)
    
    h = (b - a) / n
    print(f"Parámetros: n = {n}  ->  h = {h}")
    
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    # Tabla
    tabla = [[i, round(x[i], precision), round(y[i], precision)] for i in range(n + 1)]
    print("\nTABLA DE VALORES:")
    print(tabulate(tabla, headers=["n", "x_n", "f(x_n)"], tablefmt="grid"))
    
    # Desarrollo
    interiores = y[1:n]
    str_interiores = " + ".join([f"{val:.{precision}f}" for val in interiores])
    fraccion_h = f"{h}/2" if isinstance(h, float) else f"({b}-{a})/{2*n}"
    print(f"\nDESARROLLO: I ~= {fraccion_h} [ {y[0]:.{precision}f} + 2({str_interiores}) + {y[-1]:.{precision}f} ]")
    
    # Cálculos
    integral = (h / 2) * (y[0] + 2 * np.sum(interiores) + y[-1])
    max_segunda = np.max(np.abs(segunda_derivada_numerica(f, np.linspace(a, b, 1000))))
    cota_error = ((b - a)**3 / (12 * n**2)) * max_segunda
    
    print(f"\nRESULTADOS: I ~= {integral:.8f}  |  E_t <= ±{cota_error:.8f}")
    return integral, cota_error


def simpson_13_compuesto(f, a, b, n, precision=6):
    print("\n" + "="*60)
    print(" 3. MÉTODO: SIMPSON 1/3 COMPUESTO ")
    print("="*60)
    
    if n % 2 != 0:
        print(">> ERROR TEÓRICO: Simpson 1/3 requiere un 'n' PAR. Se omitirá este método.")
        return None, None
        
    h = (b - a) / n
    print(f"Parámetros: n = {n}  ->  h = {h}")
    
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    # Tabla
    tabla = [[i, round(x[i], precision), round(y[i], precision)] for i in range(n + 1)]
    print("\nTABLA DE VALORES:")
    print(tabulate(tabla, headers=["n", "x_n", "f(x_n)"], tablefmt="grid"))
    
    # Desarrollo
    impares = y[1:n:2]
    pares = y[2:n-1:2]
    str_impares = " + ".join([f"{val:.{precision}f}" for val in impares])
    str_pares = " + ".join([f"{val:.{precision}f}" for val in pares])
    
    termino_pares = f" + 2({str_pares})" if str_pares else ""
    fraccion_h = f"{h}/3" if isinstance(h, float) else f"({b}-{a})/{3*n}"
    print(f"\nDESARROLLO: I ~= {fraccion_h} [ {y[0]:.{precision}f} + 4({str_impares}){termino_pares} + {y[-1]:.{precision}f} ]")
    
    # Cálculos
    integral = (h / 3) * (y[0] + y[-1] + 4 * np.sum(impares) + 2 * np.sum(pares))
    max_cuarta = np.max(np.abs(cuarta_derivada_numerica(f, np.linspace(a, b, 1000))))
    cota_error = ((b - a)**5 / (180 * n**4)) * max_cuarta
    
    print(f"\nRESULTADOS: I ~= {integral:.8f}  |  E_t <= ±{cota_error:.8f}")
    return integral, cota_error


## main

if __name__ == "__main__":
    # --- PARÁMETROS GLOBALES DEL EJERCICIO ---
    a = 0
    b = 3
    n = 6 # numero de subintervalos
    decimales = 6
    
    print(f"Iniciando evaluación de la integral desde a={a} hasta b={b} con n={n} subintervalos.")

    # Diccionario para el reporte final
    reporte = []

    # 1. Ejecutar Rectángulo
    res_rect, err_rect = rectangulo_medio_compuesto(funcion, a, b, n, precision=decimales)
    # Pasamos el resultado como número puro (sin formatear como string) para que tabulate lo maneje
    reporte.append(["Rectángulo Medio", res_rect, f"±{err_rect:.{decimales}f}"])

    # 2. Ejecutar Trapecio
    res_trap, err_trap = trapecio_compuesto(funcion, a, b, n, precision=decimales)
    reporte.append(["Trapecio", res_trap, f"±{err_trap:.{decimales}f}"])

    # 3. Ejecutar Simpson 1/3
    res_simp, err_simp = simpson_13_compuesto(funcion, a, b, n, precision=decimales)
    if res_simp is not None:
        reporte.append(["Simpson 1/3", res_simp, f"±{err_simp:.{decimales}f}"])

    # --- REPORTE COMPARATIVO FINAL ---
    print("\n" + "="*60)
    print(" RESUMEN COMPARATIVO DE INTEGRACIÓN NUMÉRICA ")
    print("="*60)
    
    # Agregamos floatfmt=f".{decimales}f" para gobernar las columnas numéricas
    print(tabulate(
        reporte, 
        headers=["Método", "Integral Aproximada (I)", "Cota de Error Máximo (E_t)"], 
        tablefmt="fancy_grid", 
        floatfmt=f".{decimales}f"
    ))
    print("="*60 + "\n")