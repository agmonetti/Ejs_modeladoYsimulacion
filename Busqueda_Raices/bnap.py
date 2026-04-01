import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

'''Archivo con los 4 metodos numericos para hallar raices en uno mismo
    - Biseccion
    - Newton-Raphson
    - Punto Fijo
    - Aitken (pto fijo acelerado)

    Resultado:
    * Parametros utilizados
    * Tabla de iteraciones
    * Gráfico de la función con las raíces
    * Gráfico de Historial de Convergencia (Error vs Iteraciones)
    * Comparacion entre los distintos metodos
'''

# -----------------------------------
#  VARIABLES GLOBALES
# ------------------------------------
MAX_ITERACIONES = 100
TOLERANCIA = 1e-6
PRECISION = 10

# -----------------------------------
#  METODOS NUMERICOS
# ------------------------------------

def biseccion(f, a, b, iteraciones=MAX_ITERACIONES, tolerancia=TOLERANCIA, precision=PRECISION):
    print("\n--- INICIANDO BISECCIÓN ---")
    if f(a) * f(b) >= 0:
        raise ValueError("La función debe tener signos opuestos en los extremos del intervalo.")

    results = []
    lista_errores = []
    
    for i in range(iteraciones):
        c = (a + b) / 2.0
        fc = f(c)
        error_actual = abs(b - a) / 2.0
        
        lista_errores.append(error_actual)
        results.append([i, round(a, precision), round(b, precision), round(c, precision), round(fc, precision)])

        if abs(fc) < tolerancia or error_actual < tolerancia:
            print(tabulate(results, headers=["i", "a", "b", "c", "f(c)"]))
            return c, i + 1, lista_errores

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    print(tabulate(results, headers=["i", "a", "b", "c", "f(c)"]))
    raise ValueError("El método no convergió.")


def punto_fijo(g, x0, tol=TOLERANCIA, max_iter=MAX_ITERACIONES):
    print("\n--- INICIANDO PUNTO FIJO ---")
    x = x0
    lista_errores = []
    
    for i in range(max_iter):
        x_new = g(x)
        error_actual = abs(x_new - x)
        lista_errores.append(error_actual)
        
        print(f"Iteracion {i}: {x_new}")

        if error_actual < tol:
            print(f"Convergencia alcanzada en {i+1} iteraciones.")
            return x_new, i + 1, lista_errores
            
        x = x_new

    return None, max_iter, lista_errores


def punto_fijo_con_aitken_tabla(g, x0, tol=TOLERANCIA, max_iter=MAX_ITERACIONES):
    print("\n--- INICIANDO ACELERACIÓN DE AITKEN ---")
    x = x0
    lista_errores = []
    print(f"{'Iteración':<10}{'x':<20}{'x1 = g(x)':<20}{'x2 = g(x1)':<20}{'x_acelerado':<20}{'Error':<20}")
    print("-" * 100)
    
    for i in range(max_iter):
        x1 = g(x)
        x2 = g(x1)
        
        denominador = x2 - 2 * x1 + x
        if denominador != 0:
            x_acelerado = x - (x1 - x)**2 / denominador
        else:
            x_acelerado = x2 
            
        error_actual = abs(x_acelerado - x)
        lista_errores.append(error_actual)
        
        print(f"{i:<10}{x:<20.10f}{x1:<20.10f}{x2:<20.10f}{x_acelerado:<20.10f}{error_actual:<20.10f}")
        
        if error_actual < tol:
            print(f"\nConvergencia alcanzada en {i+1} iteraciones.")
            return x_acelerado, i + 1, lista_errores
            
        x = x_acelerado
        
    return None, max_iter, lista_errores


def derivative_for_newton_r(f, x, dx=1e-6):
    return (f(x + dx) - f(x - dx)) / (2.0 * dx)

def newton_raphson(f, valor_inicial, iteraciones=MAX_ITERACIONES, tolerancia=TOLERANCIA, precision=PRECISION):
    print("\n--- INICIANDO NEWTON-RAPHSON ---")
    x = valor_inicial
    results = []   
    lista_errores = []
    
    for i in range(iteraciones):
        fx = round(f(x), precision)
        dfx = round(derivative_for_newton_r(f, x), precision)
        
        if dfx == 0:
            raise ValueError("La derivada es cero. El método no puede continuar.")
            
        x_new = round(x - fx / dfx, precision)
        error_actual = abs(x_new - x)
        
        lista_errores.append(error_actual)
        results.append([i, x, fx, dfx, x_new, round(error_actual, precision)])
        
        if error_actual < tolerancia:
            print(tabulate(results, headers=["Iteración", "x", "f(x)", "f'(x)", "Resultado", "Error"], tablefmt="grid"))
            return x_new, i + 1, lista_errores
            
        x = x_new
        
    print(tabulate(results, headers=["Iteración", "x", "f(x)", "f'(x)", "Resultado", "Error"], tablefmt="grid"))
    raise ValueError("El método no convergió.")


# -----------------------------------
#  METODOS AUXILIARES PARA GRAFICAR
# ------------------------------------

def graficar_comparativa(f, a, b, diccionario_raices):
    """Grafica la función y superpone las raíces de todos los métodos para comparación."""
    x = np.linspace(a - 0.5, b + 0.5, 400)
    y = f(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='f(x)', color='black')
    plt.axhline(0, color='gray', linewidth=1, linestyle='--')
    
    colores = ['red', 'blue', 'green', 'orange']
    marcadores = ['o', 's', '^', 'D']
    
    for idx, (metodo, raiz) in enumerate(diccionario_raices.items()):
        if raiz is not None:
            plt.plot(raiz, f(raiz), marker=marcadores[idx], color=colores[idx], 
                     markersize=8, label=f'{metodo} (x $\\approx$ {raiz:.5f})')

    plt.grid(color='lightgray', linestyle='-', linewidth=0.5)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Comparativa de Convergencia: Métodos Numéricos')
    plt.show()


def graficar_historial_errores(diccionario_errores):
    """Genera un gráfico de decaimiento del error en escala logarítmica."""
    plt.figure(figsize=(10, 6))
    
    colores = {'Bisección': 'red', 'Punto Fijo': 'blue', 'Aitken': 'green', 'Newton-Raphson': 'orange'}
    marcadores = {'Bisección': 'o', 'Punto Fijo': 's', 'Aitken': '^', 'Newton-Raphson': 'D'}
    
    for metodo, errores in diccionario_errores.items():
        if errores: # Si la lista no está vacía
            # Eje X son los números de iteración (1, 2, 3...)
            iteraciones = list(range(1, len(errores) + 1))
            plt.plot(iteraciones, errores, marker=marcadores.get(metodo, 'o'), 
                     color=colores.get(metodo, 'black'), linestyle='-', linewidth=2, label=metodo)

    plt.yscale('log') # Magia: transforma el eje Y a logaritmo
    plt.grid(True, which="both", ls="--", linewidth=0.5) # Cuadrícula para leer logs
    plt.xlabel('Número de Iteraciones')
    plt.ylabel('Error Absoluto (Escala Logarítmica)')
    plt.title('Historial de Convergencia: Error vs. Iteraciones')
    plt.legend()
    plt.show()


# -----------------------------------
#  FUNCION F(X) y G(X)
# ------------------------------------

def f(x):
    return x - np.cos(x)

def g(x):
    return np.cos(x)


# -----------------------------------
#  MAIN: ORQUESTADOR Y COMPARATIVA
# ------------------------------------

if __name__ == "__main__":
    # Parámetros iniciales
    a = 0
    b = 1
    x0 = 0.5

    # Diccionarios para almacenar resultados para el reporte final y gráficos
    resultados_raices = {}
    resultados_comparativa = []
    historial_errores = {}

    # 1. Bisección
    try:
        raiz_bisec, iter_bisec, err_bisec = biseccion(f, a, b)
        resultados_raices["Bisección"] = raiz_bisec
        resultados_comparativa.append(["Bisección", raiz_bisec, iter_bisec])
        historial_errores["Bisección"] = err_bisec
    except Exception as e:
        print(f"Error en Bisección: {e}")

    # 2. Punto Fijo
    raiz_pf, iter_pf, err_pf = punto_fijo(g, x0)
    resultados_raices["Punto Fijo"] = raiz_pf
    resultados_comparativa.append(["Punto Fijo", raiz_pf, iter_pf])
    historial_errores["Punto Fijo"] = err_pf

    # 3. Aitken
    raiz_aitken, iter_aitken, err_aitken = punto_fijo_con_aitken_tabla(g, x0)
    resultados_raices["Aitken"] = raiz_aitken
    resultados_comparativa.append(["Aitken", raiz_aitken, iter_aitken])
    historial_errores["Aitken"] = err_aitken

    # 4. Newton-Raphson
    try:
        raiz_nr, iter_nr, err_nr = newton_raphson(f, x0)
        resultados_raices["Newton-Raphson"] = raiz_nr
        resultados_comparativa.append(["Newton-Raphson", raiz_nr, iter_nr])
        historial_errores["Newton-Raphson"] = err_nr
    except Exception as e:
        print(f"Error en Newton-Raphson: {e}")


    # --- REPORTE FINAL ---
    print("\n" + "="*50)
    print(" RESUMEN COMPARATIVO DE MÉTODOS ")
    print("="*50)
    print(tabulate(resultados_comparativa, headers=["Método", "Raíz Hallada", "Iteraciones Requeridas"], tablefmt="fancy_grid"))
    print("="*50 + "\n")

    # --- GRÁFICOS ---
    print("Generando Gráfico 1: Raíces sobre la Función f(x)...")
    graficar_comparativa(f, a, b, resultados_raices)
    
    print("Generando Gráfico 2: Historial de Convergencia...")
    graficar_historial_errores(historial_errores)