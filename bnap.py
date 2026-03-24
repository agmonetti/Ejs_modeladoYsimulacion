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
    * Grafico de la funcion con la raiz
    * Comparacion entre los distintos metodos (conclusion)
'''

# -----------------------------------
#  METODOS NUMERICOS
# ------------------------------------

''' 1) Biseccion, requiere:
        - Intervalo [a, b] ó inicio y fin para buscar intervalos
        - Funcion f(x) 
'''
def biseccion(f, a, b, iteraciones=100, tolerancia=1e-3, precision=8):
    # Verificación inicial
    if f(a) * f(b) >= 0:
        raise ValueError("La función debe tener signos opuestos en los extremos del intervalo.")

    results = []

    for i in range(iteraciones):
        c = (a + b) / 2.0
        fc = f(c)

        results.append([i, round(a, precision), round(b, precision), round(c, precision), round(fc, precision)])

        # Condición de parada
        if abs(fc) < tolerancia or (b - a) / 2.0 < tolerancia:
            # Impresión de la matriz de resultados una única vez al alcanzar la convergencia
            print(tabulate(results, headers=["i", "a", "b", "c", "f(c)"]))
            return c

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    # Impresión en caso de no convergencia para análisis de divergencia
    print(tabulate(results, headers=["i", "a", "b", "c", "f(c)"]))
    raise ValueError("El método no convergió en el número máximo de iteraciones.")
#funcion para encontrar los intervalos dado una funcion x
def buscar_intervalos(f, inicio, fin, paso=0.5):
    """
    Realiza un barrido escalar para encontrar intervalos [a, b] 
    donde la función cambia de signo (f(a) * f(b) < 0).
    """
    intervalos = []
    
    # Generamos el vector de puntos a evaluar
    puntos_x = np.arange(inicio, fin + paso, paso)
    
    for i in range(len(puntos_x) - 1):
        a = puntos_x[i]
        b = puntos_x[i+1]
        
        # Aplicación del Teorema de Bolzano
        if f(a) * f(b) < 0:
            # Guardamos el intervalo redondeado para mayor limpieza visual
            intervalos.append((round(a, 4), round(b, 4)))
            
    return intervalos



''' 2) Punto Fijo, requiere:
        - Funcion f(x) 
        - Funcion g(x) (despejo x de f(x))
        - Requiere un x0 inicial
        - Se debe cumplir condicion de lipschitz --> |g'(x0)| < 1
'''
def fixed_point_iteration(x0, tol=1e-6, max_iter=100):
    x = x0
    iter_values = [x0]

    for i in range(max_iter):
        x_new = g(x)
        iter_values.append(x_new)
        print("Iteracion " + str(i) + ": " + str(x_new))

        if abs(x_new - x) < tol:
            print("Tolerance exceeded..." )
            print(f"Converged to {x_new} after {i} iterations.")
            print(f"final error: {abs(x_new - x)}")
            break

        x = x_new

    return x_new, iter_values




''' 3) Aceleracion Aitken, requiere:
        - Funcion f(x) 
        - Funcion g(x) (despejo x de f(x))
        - Requiere un x0 inicial
        - Se debe cumplir condicion de lipschitz --> |g'(x0)| < 1
'''
def punto_fijo_con_aitken_tabla(g, x0, tol=1e-6, max_iter=100):
    x = x0
    print(f"{'Iteración':<10}{'x':<20}{'x1 = g(x)':<20}{'x2 = g(x1)':<20}{'x_acelerado':<20}{'Error':<20}")
    print("-" * 100)
    
    for i in range(max_iter):
        # Calcular tres puntos consecutivos de la iteración
        x1 = g(x)
        x2 = g(x1)
        
        # Aplicar la aceleración de Aitken
        denominador = x2 - 2 * x1 + x
        if denominador != 0:
            x_acelerado = x - (x1 - x)**2 / denominador
        else:
            x_acelerado = x2 # Si no se puede aplicar Aitken, continuar con x2
            
        # Calcular error relativo
        error = abs(x_acelerado - x)
        
        # Mostrar valores en la tabla
        print(f"{i:<10}{x:<20.10f}{x1:<20.10f}{x2:<20.10f}{x_acelerado:<20.10f}{error:<20.10f}")
        
        # Verificar convergencia
        if error < tol:
            print(f"\nConvergencia alcanzada en la iteración {i}.")
            return x_acelerado
            
        # Actualizar para la siguiente iteración
        x = x_acelerado
        
    print("\nEl método no converge después del número máximo de iteraciones.")
    return None



''' 4) Newton-Raphson, requiere:
        - Funcion f(x) 
        - Requiere un x0 inicial
        - Calcula la derivada de esa f(x)
        - Puede fallar si la derivada es cero
'''
def derivative_for_newton_r(f, x, dx=1e-6):
    """Aproximación de la derivada usando diferencias centrales"""
    return (f(x + dx) - f(x - dx)) / (2.0 * dx)

def newton_raphson(f, valor_inicial, iteraciones=100, tolerancia=1e-6, precision=10):
    x = valor_inicial
    results = []
    for i in range(iteraciones):
        fx = round(f(x), precision)
        dfx = round(derivative_for_newton_r(f, x, dx=tolerancia), precision)
        if dfx == 0:
            raise ValueError("La derivada es cero. El método no puede continuar.")
        x_new = round(x - fx / dfx, precision)
        # Calcula el error absoluto: distancia entre el valor nuevo y el anterior
        error_abs = abs(x_new - x)
        results.append([i, x, fx, dfx, x_new, round(error_abs, precision)])
        print(tabulate(results, headers=["Iteración", "x", "f(x)", "f'(x)", "Resultado", "Error"], tablefmt="grid"))
        if error_abs < tolerancia:
            return x_new
        x = x_new
    raise ValueError("El método no convergió o faltan iteraciones.")



# -----------------------------------
#  METODOS AUXILIARES PARA GRAFICAR
# ------------------------------------


## no funciona desde mi visual en arch - desde zsh, grafico visual perfectamente
def graficar_biseccion(f, a, b, raiz):
    # Graficar la función
    x = np.linspace(a - 1, b + 1, 400)
    y = f(x)
    plt.plot(x, y, label='$f(x)$')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    # Marcador analítico de la raíz
    if raiz is not None:
        plt.plot(raiz, f(raiz), 'ro', label=f'Raíz $\\approx$ {raiz:.5f}')
        
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    
    # Instrucción de renderizado y proyección de la ventana gráfica
    plt.show() 

def graficar_newton_raphson(f, raiz):
    # Graficar la función
    x = np.linspace(0, 3, 400)
    y = f(x)
    plt.plot(x, y, label='f(x)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    
    # Marcar la raíz encontrada
    plt.plot(raiz, f(raiz), 'ro', label=f'Raíz: x = {raiz:.5f}')
    
    # Añadir leyenda y mostrar la gráfica
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gráfica de la función y su raíz')
    plt.show()


# -----------------------------------
#  FUNCION F(X) y G(X)
# ------------------------------------

# f(x) para todos los metodos (menos pto fijo)
def f(x):
    return x * np.exp(-x)

# g(x) para punto fijo y aitken
def g(x):
    return  x - x*np.exp(-x)




# -----------------------------------
#  MAIN
# ------------------------------------

'''
- PARAMETROS:

'''
# Intervalo inicial Usado para BISECCION
a =-1
b =1

# punto inicial para el punto fijo y aitken
x0 = 0.5



'''
LLAMADAS A LAS FUNCIONES + GRAFICOS
'''

# Encontrar y mostrar la raíz
raiz = biseccion(f, a, b)
print(f"La raíz encontrada es: {raiz}")
print("Graficando...")
graficar_biseccion(f, a, b, raiz)

#pto fijo
root, iter_values = fixed_point_iteration(x0)

# aitken
resultado = punto_fijo_con_aitken_tabla(g, x0)
if resultado is not None:
 print(f"\nLa solución aproximada es: {resultado}")

 # Encontrar la raíz utilizando el método de Newton-Raphson
raiz_raphson = newton_raphson(f, x0)
# Imprimir la raíz encontrada
print(f"La raíz encontrada es: {raiz_raphson}")
# Graficar la función y la raíz
graficar_newton_raphson(f, raiz_raphson)