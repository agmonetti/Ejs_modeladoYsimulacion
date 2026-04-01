import numpy as np
import warnings
from tabulate import tabulate
import matplotlib.pyplot as plt

'''
Archivo Consolidador: Métodos de Integración Numérica (Fórmulas de Newton-Cotes)
    - Regla del Rectángulo Medio Compuesto
    - Regla del Trapecio Compuesto
    - Regla de Simpson 1/3 Compuesto
    - Regla de Simpson 3/8 Compuesto

    Resultado por cada método:
    * Parámetros utilizados (a, b, n, h)
    * Tabla de iteraciones/evaluaciones (Formato Pizarra UADE)
    * Desarrollo algebraico de la sumatoria
    * Cálculo de la integral aproximada
    * Cálculo de la cota máxima de Error de Truncamiento (E_t)
'''

# -----------------------------------
#  FUNCION A INTEGRAR
# -----------------------------------
def funcion(x):
    numerador = (2 + np.cos(1 + x**(3 / 2))) 
    denominador = (1 + 0.5 * np.sin(x)) ** (1/2)
    return (numerador / denominador)* np.exp(0.5 * x)


# -----------------------------------
#  EL ESCUDO NUMÉRICO (Auto-Reparación)
# -----------------------------------
def evaluar_seguro(f, x, silencioso=False):
    """
    Evalúa cualquier función f en el vector x. 
    Si encuentra un NaN o Infinito, lo repara por límite y avisa.
    """
    x_arr = np.atleast_1d(x).astype(float)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = f(x_arr)
        
    malos_indices = np.where(np.isnan(y) | np.isinf(y))[0]
    
    if len(malos_indices) > 0:
        if not silencioso:
            print("Se detectó una indeterminación (NaN/Infinito) al evaluar la función.")
            
        epsilon = 1e-9 
        for i in malos_indices:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y[i] = f(x_arr[i] + epsilon)
            if not silencioso:
                print(f" -> El punto crítico x = {x_arr[i]} fue aproximado por límite a f(x) ~= {y[i]:.6f}")
                
    return y[0] if len(y) == 1 else y


# -----------------------------------
#  DERIVADAS NUMÉRICAS (Sigilosas)
# -----------------------------------
def segunda_derivada_numerica(f, x, dx=1e-5):
    f_xdx = evaluar_seguro(f, x + dx, silencioso=True)
    f_x   = evaluar_seguro(f, x, silencioso=True)
    f_mdx = evaluar_seguro(f, x - dx, silencioso=True)
    return (f_xdx - 2 * f_x + f_mdx) / (dx**2)

def cuarta_derivada_numerica(f, x, dx=1e-3):
    f_x2dx = evaluar_seguro(f, x + 2*dx, silencioso=True)
    f_xdx  = evaluar_seguro(f, x + dx, silencioso=True)
    f_x    = evaluar_seguro(f, x, silencioso=True)
    f_mdx  = evaluar_seguro(f, x - dx, silencioso=True)
    f_m2dx = evaluar_seguro(f, x - 2*dx, silencioso=True)
    return (f_x2dx - 4*f_xdx + 6*f_x - 4*f_mdx + f_m2dx) / (dx**4)


# -----------------------------------
#  MÉTODOS NUMÉRICOS
# -----------------------------------

def rectangulo_medio_compuesto(f, a, b, n, precision=6):
    print("\n" + "="*60)
    print(" 1. MÉTODO: RECTÁNGULO MEDIO COMPUESTO ")
    print("="*60)
    
    h = (b - a) / n
    print(f"Parámetros: n = {n}  ->  h = {h}")
    
    x = np.linspace(a, b, n + 1)
    x_medio = np.linspace(a + h/2, b - h/2, n)
    y_medio = evaluar_seguro(f, x_medio)
    
    tabla = [[0, round(x[0], precision), "-", "-"]]
    for i in range(1, n + 1):
        tabla.append([i, round(x[i], precision), round(x_medio[i-1], precision), round(y_medio[i-1], precision)])
        
    print("\nTABLA DE VALORES:")
    print(tabulate(tabla, headers=["n", "x_n", "x_medio_n", "f(x_medio_n)"], tablefmt="grid", disable_numparse=True))
    
    str_valores = " + ".join([f"{val:.{precision}f}" for val in y_medio])
    print(f"\nDESARROLLO: A ~= {h} [ {str_valores} ]")
    
    integral = h * np.sum(y_medio)
    
    dx_seg = 1e-3
    x_fino = np.linspace(a + 2*dx_seg, b - 2*dx_seg, 1000)
    max_segunda = np.max(np.abs(segunda_derivada_numerica(f, x_fino)))
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
    y = evaluar_seguro(f, x)
    
    tabla = [[i, round(x[i], precision), round(y[i], precision)] for i in range(n + 1)]
    print("\nTABLA DE VALORES:")
    print(tabulate(tabla, headers=["n", "x_n", "f(x_n)"], tablefmt="grid", disable_numparse=True))
    
    interiores = y[1:n]
    str_interiores = " + ".join([f"{val:.{precision}f}" for val in interiores])
    fraccion_h = f"{h}/2" if isinstance(h, float) else f"({b}-{a})/{2*n}"
    print(f"\nDESARROLLO: I ~= {fraccion_h} [ {y[0]:.{precision}f} + 2({str_interiores}) + {y[-1]:.{precision}f} ]")
    
    integral = (h / 2) * (y[0] + 2 * np.sum(interiores) + y[-1])
    
    dx_seg = 1e-3
    x_fino = np.linspace(a + 2*dx_seg, b - 2*dx_seg, 1000)
    max_segunda = np.max(np.abs(segunda_derivada_numerica(f, x_fino)))
    cota_error = ((b - a)**3 / (12 * n**2)) * max_segunda
    
    print(f"\nRESULTADOS: I ~= {integral:.8f}  |  E_t <= ±{cota_error:.8f}")
    return integral, cota_error


def simpson_13_compuesto(f, a, b, n, precision=6):
    print("\n" + "="*60)
    print(" 3. MÉTODO: SIMPSON 1/3 COMPUESTO ")
    print("="*60)
    
    if n % 2 != 0:
        print(f">> ERROR TEÓRICO: Simpson 1/3 requiere un 'n' PAR (recibió n={n}). Se omitirá este método.")
        return None, None
        
    h = (b - a) / n
    print(f"Parámetros: n = {n}  ->  h = {h}")
    
    x = np.linspace(a, b, n + 1)
    y = evaluar_seguro(f, x)
    
    tabla = [[i, round(x[i], precision), round(y[i], precision)] for i in range(n + 1)]
    print("\nTABLA DE VALORES:")
    print(tabulate(tabla, headers=["n", "x_n", "f(x_n)"], tablefmt="grid", disable_numparse=True))
    
    impares = y[1:n:2]
    pares = y[2:n-1:2]
    str_impares = " + ".join([f"{val:.{precision}f}" for val in impares])
    str_pares = " + ".join([f"{val:.{precision}f}" for val in pares])   
    
    termino_pares = f" + 2({str_pares})" if str_pares else ""
    fraccion_h = f"{h}/3" if isinstance(h, float) else f"({b}-{a})/{3*n}"
    print(f"\nDESARROLLO: I ~= {fraccion_h} [ {y[0]:.{precision}f} + 4({str_impares}){termino_pares} + {y[-1]:.{precision}f} ]")
    
    integral = (h / 3) * (y[0] + y[-1] + 4 * np.sum(impares) + 2 * np.sum(pares))
    
    dx_seg = 1e-3
    x_fino = np.linspace(a + 2*dx_seg, b - 2*dx_seg, 1000)
    max_cuarta = np.max(np.abs(cuarta_derivada_numerica(f, x_fino)))
    cota_error = ((b - a)**5 / (180 * n**4)) * max_cuarta
    
    print(f"\nRESULTADOS: I ~= {integral:.8f}  |  E_t <= ±{cota_error:.8f}")
    return integral, cota_error


def simpson_38_compuesto(f, a, b, n, precision=6):
    print("\n" + "="*60)
    print(" 4. MÉTODO: SIMPSON 3/8 COMPUESTO ")
    print("="*60)
    
    if n % 3 != 0:
        print(f">> ERROR TEÓRICO: Simpson 3/8 exige que 'n' sea MÚLTIPLO DE 3 (recibió n={n}). Se omitirá este método.")
        return None, None

    h = (b - a) / n
    fraccion_display = f"{int(b-a)}/{n}" if (b-a).is_integer() else f"{h:.{precision}f}"
    print(f"Parámetros: n = {n}  ->  h = {fraccion_display}")
    
    x = np.linspace(a, b, n + 1)
    y = evaluar_seguro(f, x)
    
    tabla = []
    for i in range(n + 1):
        x_str = f"{i}/{n}" if a == 0 and b == 1 else f"{round(x[i], precision)}"
        tabla.append([i, x_str, round(y[i], precision)])
        
    print("\nTABLA DE VALORES:")
    print(tabulate(tabla, headers=["n", "x_n", "f(x_n)"], tablefmt="grid", disable_numparse=True))
    
    grupo_1 = y[1:n:3]
    grupo_2 = y[2:n:3]
    grupo_3 = y[3:n-1:3]
    
    str_1 = " + ".join([f"{val:.{precision}f}" for val in grupo_1])
    str_2 = " + ".join([f"{val:.{precision}f}" for val in grupo_2])
    str_3 = " + ".join([f"{val:.{precision}f}" for val in grupo_3])
    
    fraccion_h = f"3({fraccion_display})/8"
    termino_mult3 = f" + 2({str_3})" if str_3 else ""
    desarrollo = f"I ~= {fraccion_h} [ {y[0]:.{precision}f} + 3({str_1}) + 3({str_2}){termino_mult3} + {y[-1]:.{precision}f} ]"
    
    print("\nDESARROLLO DE LA FÓRMULA:")
    print(desarrollo)
    
    S = y[0] + y[-1] + 3 * np.sum(grupo_1) + 3 * np.sum(grupo_2) + 2 * np.sum(grupo_3)
    integral = (3 * h / 8) * S
    
    dx_seg = 1e-3
    x_fino = np.linspace(a + 2*dx_seg, b - 2*dx_seg, 1000) 
    max_cuarta = np.max(np.abs(cuarta_derivada_numerica(f, x_fino)))
    cota_error = ((b - a)**5 / (80 * n**4)) * max_cuarta
    
    print(f"\nRESULTADOS FINALES:")
    print(f"I ~= {integral:.8f}  |  E_t <= ±{cota_error:.8f}")
    return integral, cota_error

# -----------------------------------
#  MÉTODO AUXILIAR PARA GRAFICAR
# -----------------------------------
def graficar_integral(f, a, b):
    print("Generando Gráfico de la función y el área a integrar...")
    
    # Generamos un margen del 20% para que no quede pegado a los bordes
    margen = (b - a) * 0.2 if a != b else 1
    
    # 1. Puntos para la curva completa (usamos evaluar_seguro en modo silencioso)
    x_plot = np.linspace(a - margen, b + margen, 400)
    y_plot = evaluar_seguro(f, x_plot, silencioso=True)
    
    # 2. Puntos específicos para el sombreado del área
    x_area = np.linspace(a, b, 400)
    y_area = evaluar_seguro(f, x_area, silencioso=True)
    
    # Configuración del lienzo
    plt.figure(figsize=(10, 6))
    
    # Dibujamos la función
    plt.plot(x_plot, y_plot, 'k-', linewidth=2, label='f(x)')
    
    # Rellenamos el área bajo la curva
    plt.fill_between(x_area, y_area, alpha=0.3, color='blue', label=f'Área integral [{a}, {b}]')
    
    # Líneas verticales para marcar los límites exactos
    plt.axvline(x=a, color='red', linestyle='--', alpha=0.7, label=f'Límite a={a}')
    plt.axvline(x=b, color='green', linestyle='--', alpha=0.7, label=f'Límite b={b}')
    plt.axhline(0, color='black', linewidth=0.5) # Eje X
    
    plt.title('Representación Gráfica de la Integral Numérica')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.show()


## main
if __name__ == "__main__":
    # --- PARÁMETROS GLOBALES DEL EJERCICIO ---
    a = 0
    b = 2
    n = 4 # numero de subintervalos
    decimales = 6
    
    print(f"Iniciando evaluación de la integral desde a={a} hasta b={b} con n={n} subintervalos.")

    # Diccionario para el reporte final
    reporte = []

    # 1. Ejecutar Rectángulo
    res_rect, err_rect = rectangulo_medio_compuesto(funcion, a, b, n, precision=decimales)
    reporte.append(["Rectángulo Medio", res_rect, f"±{err_rect:.{decimales}f}"])

    # 2. Ejecutar Trapecio
    res_trap, err_trap = trapecio_compuesto(funcion, a, b, n, precision=decimales)
    reporte.append(["Trapecio", res_trap, f"±{err_trap:.{decimales}f}"])

    # 3. Ejecutar Simpson 1/3
    res_simp, err_simp = simpson_13_compuesto(funcion, a, b, n, precision=decimales)
    if res_simp is not None:
        reporte.append(["Simpson 1/3", res_simp, f"±{err_simp:.{decimales}f}"])

    # 4. Ejecutar Simpson 3/8
    res_simp38, err_simp38 = simpson_38_compuesto(funcion, a, b, n, precision=decimales)
    if res_simp38 is not None:
        reporte.append(["Simpson 3/8", res_simp38, f"±{err_simp38:.{decimales}f}"])

    # --- REPORTE COMPARATIVO FINAL ---
    print("\n" + "="*60)
    print(" RESUMEN COMPARATIVO DE INTEGRACIÓN NUMÉRICA ")
    print("="*60)
    
    print(tabulate(
        reporte, 
        headers=["Método", "Integral Aproximada (I)", "Cota de Error Máximo (E_t)"], 
        tablefmt="fancy_grid", 
        floatfmt=f".{decimales}f"
    ))
    print("="*60 + "\n")
    
    # --- GRÁFICA FINAL ---
    graficar_integral(funcion, a, b)