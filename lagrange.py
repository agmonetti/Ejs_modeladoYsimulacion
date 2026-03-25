import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# ==========================================
# calculos
# ==========================================

def calcular_polinomio_lagrange(func_str, puntos_x, x_eval):
    """Genera el polinomio de Lagrange, muestra los pasos y lo evalúa en x_eval."""
    x = sp.Symbol('x')
    f = sp.sympify(func_str)
    n = len(puntos_x)
    
    # Calcular imágenes
    puntos_y = [f.subs(x, xi) for xi in puntos_x]
    
    print("--- 1. Puntos Evaluados ---")
    for xi, yi in zip(puntos_x, puntos_y):
        print(f"x = {float(xi):.4f}  =>  y = f(x) = {float(yi):.6f}")
    
    print("\n--- 2. Construcción del Polinomio ---")
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
    
    # Evaluación explícita que pediste
    P_eval_num = float(P_final.subs(x, x_eval).evalf())
    print(f"\n=> P_{n-1}({x_eval}) ~= {P_eval_num:.6f}\n")
    
    return f, P_final, puntos_y, P_eval_num

def calcular_error_local(f_expr, P_expr, x_eval, P_eval_num):
    """Calcula la diferencia exacta entre la función y el polinomio en un punto."""
    x = sp.Symbol('x')
    f_eval = float(f_expr.subs(x, x_eval).evalf())
    error_local = abs(f_eval - P_eval_num)
    
    print("--- 3. Error Local ---")
    print(f"f({x_eval}) = {f_eval:.6f}")
    print(f"P({x_eval}) = {P_eval_num:.6f}")
    print(f"Error Local = |f({x_eval}) - P({x_eval})| = {error_local:.6f}\n")
    
    return error_local

def calcular_cota_global(f_expr, puntos_x):
    """Calcula la cota teórica máxima del error en el intervalo."""
    x = sp.Symbol('x')
    n = len(puntos_x)
    x_min, x_max = float(min(puntos_x)), float(max(puntos_x))
    
    print("--- 4. Cota de Error Global ---")
    derivada_n = sp.diff(f_expr, x, n) 
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
    
    puntos_evaluar_g = [x_min, x_max] + [float(r) for r in raices_validas]
    max_g = max([abs(float(g.subs(x, val).evalf())) for val in puntos_evaluar_g])
    
    fact_n = sp.factorial(n)
    cota_global = (max_der / fact_n) * max_g
    
    print(f"Máximo de la derivada en el intervalo ~= {max_der:.6f}")
    print(f"Máximo de |g(x)| en el intervalo ~= {max_g:.6f}")
    print(f"Cota de Error Global = {float(cota_global):.6f}\n")
    
    return cota_global

# ==========================================
# grafico
# ==========================================

def graficar_interpolacion(func_str, f_expr, P_expr, puntos_x, puntos_y, x_eval, P_eval_num):
    """Renderiza el gráfico comparativo."""
    x = sp.Symbol('x')
    x_min, x_max = float(min(puntos_x)), float(max(puntos_x))
    
    f_num = sp.lambdify(x, f_expr, 'numpy')
    P_num = sp.lambdify(x, P_expr, 'numpy')
    
    x_plot = np.linspace(x_min - 0.5, x_max + 0.5, 400)
    y_f = f_num(x_plot)
    y_P = P_num(x_plot)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_f, label='f(x) Original', linestyle='--', color='blue', alpha=0.7)
    plt.plot(x_plot, y_P, label='P(x) Lagrange', color='orange')
    plt.scatter([float(xi) for xi in puntos_x], [float(yi) for yi in puntos_y], color='red', zorder=5, label='Puntos base')
    plt.scatter([float(x_eval)], [P_eval_num], color='green', marker='X', s=100, zorder=5, label=f'Eval (x={float(x_eval):.2f})')
    
    plt.title(f'Interpolación: f(x) = {func_str}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

# ==========================================
# ORQUESTADOR
# ==========================================

def ejecutar_ejercicio(func_str, puntos_x, x_eval):
    """Coordina la ejecución de todos los módulos para un ejercicio."""
    print("="*60)
    print(f" RESOLVIENDO: f(x) = {func_str} ")
    print("="*60, "\n")
    
    # 1. Armar Polinomio y evaluarlo
    f_expr, P_expr, puntos_y, P_eval_num = calcular_polinomio_lagrange(func_str, puntos_x, x_eval)
    
    # 2. Calcular Errores
    error_local = calcular_error_local(f_expr, P_expr, x_eval, P_eval_num)
    cota_global = calcular_cota_global(f_expr, puntos_x)
    
    # 3. Demostración
    print("--- 5. Demostración Final ---")
    if error_local <= cota_global:
        print(f"¡Éxito! Cota global ({float(cota_global):.6f}) >= Error Local ({error_local:.6f})")
    else:
        print(f"ATENCIÓN: Cota global ({float(cota_global):.6f}) < Error Local ({error_local:.6f}). Revisar.")
        
    # 4. Graficar
    graficar_interpolacion(func_str, f_expr, P_expr, puntos_x, puntos_y, x_eval, P_eval_num)


# ==========================================
# main
# ==========================================

# Ejercicio 1
#ejecutar_ejercicio('exp(x)', [1, 2, 3], 1.3)

# Ejercicio 2 (Descomentalo cuando quieras probar el de seno)
ejecutar_ejercicio('sin(x)', [0, np.pi/2, np.pi], np.pi/4)