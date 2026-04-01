import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# ==========================================
# calculos
# ==========================================

def calcular_polinomio_lagrange(puntos_x, x_eval, func_str=None, puntos_y=None):
    """Genera el polinomio de Lagrange, muestra los pasos, verifica nodos y evalúa en x_eval.
    
    Parámetros:
    - puntos_x: Lista de puntos x para interpolación
    - x_eval: Punto donde evaluar el polinomio
    - func_str: (Opcional) String de la función f(x). Si se proporciona, se calcula puntos_y
    - puntos_y: (Opcional) Lista de puntos y directa. Si func_str no se proporciona, usar esto
    """
    x = sp.Symbol('x')
    n = len(puntos_x)
    f = None
    
    # Determinar si usamos función explícita o datos directos
    if func_str is not None:
        f = sp.sympify(func_str)
        # Calcular imágenes a partir de la función
        puntos_y = [f.subs(x, xi) for xi in puntos_x]
    elif puntos_y is not None:
        # Usar los puntos_y proporcionados directamente
        if len(puntos_y) != len(puntos_x):
            raise ValueError("puntos_x y puntos_y deben tener la misma longitud")
    else:
        raise ValueError("Debe proporcionar func_str o puntos_y")
    
    print("--- 1. Puntos Evaluados ---")
    print()
    for xi, yi in zip(puntos_x, puntos_y):
        print(f"x = {float(xi):.4f}  =>  y = {float(yi):.6f}")
    
    print("\n--- 2. Construcción del Polinomio ---")
    print()
    P = 0
    for i in range(n):
        if puntos_y[i] == 0:
            print(f"Demostración: Como y_{i} = 0, el término l_{i}(x) se anula.")
            continue
            
        li = 1
        for j in range(n):
            if i != j:
                li *= (x - puntos_x[j]) / (puntos_x[i] - puntos_x[j])
        
        li = sp.simplify(li)
        termino = puntos_y[i] * li
        P += termino
        print(f"l_{i}(x) * y_{i} = {sp.expand(termino)}")
        
    P_final = sp.expand(P)
    print(f"\nPolinomio final aproximado P(x) = \n{P_final}")
    

    # Evaluación explícita solicitada
    if x_eval != 999:
        P_eval_num = float(P_final.subs(x, x_eval).evalf())
        print(f"\n=> Evaluación: P_{n-1}({x_eval}) ~= {P_eval_num:.6f}\n")
    else:
        P_eval_num = x_eval

    
    return f, P_final, puntos_y, P_eval_num

def calcular_error_local(f_expr, P_expr, x_eval, P_eval_num):
    """Calcula la diferencia exacta entre la función y el polinomio en un punto."""
    x = sp.Symbol('x')
    print(f"--- 3. Error local en {x_eval} ---")
    print()
    
    if f_expr is None:
        print(f"P({x_eval}) = {P_eval_num:.6f}")
        print("(No disponible error local: función original no proporcionada)\n")
        return None
    
    f_eval = float(f_expr.subs(x, x_eval).evalf())
    error_local = abs(f_eval - P_eval_num)
    
    print(f"f({x_eval}) = {f_eval:.6f}")
    print(f"P({x_eval}) = {P_eval_num:.6f}")
    print(f"Error Local = |f({x_eval}) - P({x_eval})| = {error_local:.6f}\n")
    
    return error_local

def calcular_cota_global(f_expr, puntos_x):
    """Calcula la cota teórica máxima del error en el intervalo."""
    if f_expr is None:
        print("--- 4. Cota de Error Global ---")
        print()
        print("(No disponible: función original no proporcionada)\n")
        return None
    
    x = sp.Symbol('x')
    n = len(puntos_x)
    x_min, x_max = float(min(puntos_x)), float(max(puntos_x))
    
    print("--- 4. Cota de Error Global---")
    print()
    derivada_n = sp.diff(f_expr, x, n) 
    print(f"Necesito la derivada de orden {n} porque el polinomio es de grado {n-1} (hay {n} puntos).")
    print(f"Derivada {n} de f(x) = {derivada_n}")
    
    # Máximo de la derivada
    intervalo_x = np.linspace(x_min, x_max, 1000)
    f_der_func = sp.lambdify(x, sp.Abs(derivada_n), 'numpy')
    max_der = np.max(f_der_func(intervalo_x))
    
    # Productoria g(x)
    g = 1
    for xi in puntos_x:
        g *= (x - xi)
    g = sp.expand(g)
    g_prima = sp.diff(g, x)
    
    # Raíces de g'(x)
    raices_g_prima = sp.solve(g_prima, x)
    raices_validas = [r for r in raices_g_prima if r.is_real and x_min <= float(r) <= x_max]
    
    # Maximo de |g(x)| en el intervalo
    puntos_evaluar_g = [x_min, x_max] + [float(r) for r in raices_validas]
    max_g = max([abs(float(g.subs(x, val).evalf())) for val in puntos_evaluar_g])
    
    fact_n = sp.factorial(n)
    cota_global = (max_der / fact_n) * max_g
    
    print(f"Máximo de la derivada en el intervalo ~= {max_der:.6f}")
    print(f"Máximo de |g(x)| en el intervalo ~= {max_g:.6f}")
    print(f"Cota de Error Global ->  ({max_der:.6f} / {fact_n}) * {max_g:.6f} = {float(cota_global):.6f}\n")
    
    return cota_global

# ==========================================
# grafico
# ==========================================

def graficar_interpolacion(func_str, f_expr, P_expr, puntos_x, puntos_y, x_eval, P_eval_num):
    """Renderiza el gráfico comparativo."""
    x = sp.Symbol('x')
    x_min, x_max = float(min(puntos_x)), float(max(puntos_x))
    
    P_num = sp.lambdify(x, P_expr, 'numpy')
    
    x_plot = np.linspace(x_min - 0.5, x_max + 0.5, 400)
    y_P = P_num(x_plot)
    
    plt.figure(figsize=(10, 6))
    
    if f_expr is not None:
        # Graficar ambas curvas
        f_num = sp.lambdify(x, f_expr, 'numpy')
        y_f = f_num(x_plot)
        plt.plot(x_plot, y_f, label='f(x) Original', linestyle='--', color='blue', alpha=0.7)
        title = f'Interpolación: f(x) = {func_str}'
    else:
        # Solo graficar el polinomio interpolador
        title = 'Interpolación: Datos Directos'
    
    plt.plot(x_plot, y_P, label='P(x) Lagrange', color='orange')
    plt.scatter([float(xi) for xi in puntos_x], [float(yi) for yi in puntos_y], color='red', zorder=5, label='Puntos base')
    plt.scatter([float(x_eval)], [P_eval_num], color='green', marker='X', s=100, zorder=5, label=f'Eval (x={float(x_eval):.2f})')
    
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

# ==========================================
# ORQUESTADOR
# ==========================================

def ejecutar_ejercicio(puntos_x, x_eval, func_str=None, puntos_y=None):
    """Coordina la ejecución de todos los módulos para un ejercicio."""
    print()
    print("="*60)
    if func_str is not None:
        print(f" RESOLVIENDO: f(x) = {func_str} ")
    else:
        print(f" RESOLVIENDO: Interpolación con datos directos ")
    print("="*60, "\n")
    

    # 1, 2 y 3. Armar Polinomio, evaluar nodos y punto pedido
    f_expr, P_expr, puntos_y_calc, P_eval_num = calcular_polinomio_lagrange(puntos_x, x_eval, func_str=func_str, puntos_y=puntos_y)
    
    # 4. Calcular Errores
    if(P_eval_num != x_eval):
        error_local = calcular_error_local(f_expr, P_expr, x_eval, P_eval_num)
    
         # 5. Cota global
        cota_global = calcular_cota_global(f_expr, puntos_x)
    
        # 6. Demostración
        print("--- 5. Demostración Final ---")
        if error_local is not None and cota_global is not None:
            if error_local <= cota_global:
                print()
                print(f"¡Éxito! Cota global ({float(cota_global):.6f}) >= Error Local ({error_local:.6f})")
                print()
            else:
                print()
                print(f"ATENCIÓN: Cota global ({float(cota_global):.6f}) < Error Local ({error_local:.6f}). Revisar.")
                print()
        else:
            print()
            print("(Demostración de cota omitida: faltan datos de la función original)")
            print()
            
        # 7. Graficar
        graficar_interpolacion(func_str, f_expr, P_expr, puntos_x, puntos_y_calc, x_eval, P_eval_num)

    else:
        print()
        print(f"Al ser el pto de evaluacion {x_eval}, solo se construye el polinomio")

# ==========================================
# main
# ==========================================

# Firma de la función: (puntos_x, x_eval, func_str=None, puntos_y=None)
# si x_eval = 999 -> solo se calcula el polinomio

# CASO 1: Con función explícita
# ejecutar_ejercicio([1, 2, 3], 1.3, func_str='exp(x)')

# CASO 2: Con datos directos
# ejecutar_ejercicio([0, 1, 3, 4], 0.45, puntos_y=[1.0, 2.718, 20.086, 54.598])

ejecutar_ejercicio([1, 2, 3], 1.3, func_str="exp(x)")