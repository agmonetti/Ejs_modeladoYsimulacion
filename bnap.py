import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


def f(x):
    return x**6 -2

def g(x):
    return np.log(x + 2)

def derivative(f, x, dx=1e-6):
    """Aproximación de la derivada usando diferencias centrales"""
    return (f(x + dx) - f(x - dx)) / (2.0 * dx)


# =========================================================
# IMPLEMENTACIÓN DE LOS MÉTODOS
# Basados en los scripts originales, estandarizados para análisis
# =========================================================

def biseccion(f, a, b, iteraciones=100, tolerancia=1e-6):
    if f(a) * f(b) >= 0:
        raise ValueError("La función debe tener signos opuestos en los extremos del intervalo.")

    results = []
    errores = []

    for i in range(iteraciones):
        c = (a + b) / 2.0
        fc = f(c)
        error = (b - a) / 2.0

        results.append([i, a, b, c, fc, error])
        errores.append(error)

        if abs(fc) < tolerancia or error < tolerancia:
            return c, i, results, errores

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    return c, iteraciones - 1, results, errores


def newton_raphson(f, valor_inicial, iteraciones=100, tolerancia=1e-6):
    x = valor_inicial
    results = []
    errores = []
    
    for i in range(iteraciones):
        fx = f(x)
        dfx = derivative(f, x, dx=tolerancia)
        
        if dfx == 0:
            raise ValueError("La derivada es cero. El método no puede continuar.")
            
        x_new = x - fx / dfx
        error_abs = abs(x_new - x)
        
        results.append([i, x, fx, dfx, x_new, error_abs])
        errores.append(error_abs)
        
        if error_abs < tolerancia:
            return x_new, i, results, errores
            
        x = x_new
        
    return x, iteraciones - 1, results, errores


def punto_fijo(g, x0, tol=1e-6, max_iter=100):
    x = x0
    results = []
    errores = []

    for i in range(max_iter):
        x_new = g(x)
        error = abs(x_new - x)
        
        results.append([i, x, x_new, error])
        errores.append(error)

        if error < tol:
            return x_new, i, results, errores

        x = x_new

    return x, max_iter - 1, results, errores


def punto_fijo_aitken(g, x0, tol=1e-6, max_iter=100):
    x = x0
    results = []
    errores = []
    
    for i in range(max_iter):
        x1 = g(x)
        x2 = g(x1)
        denominador = x2 - 2 * x1 + x
        
        if denominador != 0:
            x_acelerado = x - (x1 - x)**2 / denominador
        else:
            x_acelerado = x2
            
        error = abs(x_acelerado - x)
        results.append([i, x, x1, x2, x_acelerado, error])
        errores.append(error)
        
        if error < tol:
            return x_acelerado, i, results, errores
            
        x = x_acelerado
        
    return x, max_iter - 1, results, errores


# =========================================================
# EJECUCIÓN Y ORQUESTACIÓN CENTRALIZADA
# =========================================================

# Parámetros Globales
tolerancia_global = 1e-6
max_iteraciones = 100

# Parámetros Específicos por método
a_bis, b_bis = 0, 2
#x0_newton_raphson = 1
#x0_punto_fijo = 1
#x0_aitken = 1

x0_global = 1

# --- 1. BISECCIÓN ---
raiz_bis, iter_bis, res_bis, err_bis = biseccion(f, a_bis, b_bis, iteraciones=max_iteraciones, tolerancia=tolerancia_global)

# --- 2. NEWTON RAPHYSON ---
raiz_nr, iter_nr, res_nr, err_nr = newton_raphson(f, x0_global, iteraciones=max_iteraciones, tolerancia=tolerancia_global)

# --- 3. PUNTO FIJO ---
raiz_pf, iter_pf, res_pf, err_pf = punto_fijo(g, x0_global, tol=tolerancia_global, max_iter=max_iteraciones)

# --- 4. AITKEN ---
raiz_aitken, iter_aitken, res_aitken, err_aitken = punto_fijo_aitken(g, x0_global, tol=tolerancia_global, max_iter=max_iteraciones)


# =========================================================
# IMPRESIÓN Y ANÁLISIS DE RESULTADOS
# =========================================================

print("============= RESOLUCIÓN UNIFICADA MÉTODOS NUMÉRICOS =============")
print(f"Función Evaluada: f(x) = exp(x) - 2 - x = 0")
print(f"Despeje Punto Fijo: g(x) = ln(x + 2)")
print(f"Tolerancia Objetivo: {tolerancia_global}\n")

print("--- Método de Bisección ---")
print(f"Intervalo inicial: [{a_bis}, {b_bis}]")
print(tabulate(res_bis, headers=["Iteración", "a", "b", "c", "f(c)", "Error"], floatfmt=".10f", tablefmt="grid"))

print("\n--- Método de Newton-Raphson ---")
print(f"Valor inicial (x0): {x0_global}")
print(tabulate(res_nr, headers=["Iteración", "x", "f(x)", "f'(x)", "x_nuevo", "Error"], floatfmt=".10f", tablefmt="grid"))

print("\n--- Método de Punto Fijo ---")
print(f"Valor inicial (x0): {x0_global}")
print(tabulate(res_pf, headers=["Iteración", "x", "x_nuevo", "Error"], floatfmt=".10f", tablefmt="grid"))

print("\n--- Método de Punto Fijo (Aceleración de Aitken) ---")
print(f"Valor inicial (x0): {x0_global}")
print(tabulate(res_aitken, headers=["Iteración", "x", "x1 = g(x)", "x2 = g(x1)", "x_acelerado", "Error"], floatfmt=".10f", tablefmt="grid"))


print("\n============= TABLA GLOBAL COMPARATIVA =============")
tabla_comparativa = [
    ["Bisección", raiz_bis, iter_bis, err_bis[-1]],
    ["Newton-Raphson", raiz_nr, iter_nr, err_nr[-1]],
    ["Punto Fijo", raiz_pf, iter_pf, err_pf[-1]],
    ["Punto Fijo (Aitken)", raiz_aitken, iter_aitken, err_aitken[-1]]
]
print(tabulate(tabla_comparativa, headers=["Método", "Raíz Encontrada", "Iteraciones Tomadas", "Error Final Alcanzado"], floatfmt=".10f", tablefmt="grid"))


# =========================================================
# GRÁFICOS COMPARATIVOS
# =========================================================

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico 1: Visualización de la Función y todas las Raíces Encontradas
x_vals = np.linspace(-1, 3, 400)
y_vals = f(x_vals)
axs[0].plot(x_vals, y_vals, label='$f(x) = e^x - 2 - x$', color='purple', zorder=1)
axs[0].axhline(0, color='black', linewidth=0.5)
axs[0].axvline(0, color='black', linewidth=0.5)

# Ploteamos cada raíz con un tamaño/forma ligeramente distinto para que se puedan ver si se solapan
if raiz_bis is not None:
    axs[0].plot(raiz_bis, f(raiz_bis), 'o', markersize=10, color='tab:blue', alpha=0.6, label=f'Bisección: {raiz_bis:.6f}')
if raiz_nr is not None:
    axs[0].plot(raiz_nr, f(raiz_nr), 's', markersize=8, color='tab:orange', alpha=0.7, label=f'Newton-Raphson: {raiz_nr:.6f}')
if raiz_pf is not None:
    axs[0].plot(raiz_pf, f(raiz_pf), '^', markersize=6, color='tab:green', alpha=0.8, label=f'Punto Fijo: {raiz_pf:.6f}')
if raiz_aitken is not None:
    axs[0].plot(raiz_aitken, f(raiz_aitken), 'd', markersize=4, color='tab:red', alpha=1.0, label=f'Aitken: {raiz_aitken:.6f}')

axs[0].set_title('Función $f(x)$ y convergencia de las Raíces')
axs[0].set_xlabel('x')
axs[0].set_ylabel('f(x)')
axs[0].grid(color='gray', linestyle='--', linewidth=0.5)
axs[0].legend()

# Gráfico 2: Análisis de Convergencia (Error Vs Iteraciones) para los 4 métodos
axs[1].plot(range(len(err_bis)), err_bis, marker='o', label='Bisección', color='tab:blue')
axs[1].plot(range(len(err_nr)), err_nr, marker='s', label='Newton-Raphson', color='tab:orange')
axs[1].plot(range(len(err_pf)), err_pf, marker='^', label='Punto Fijo', color='tab:green')
axs[1].plot(range(len(err_aitken)), err_aitken, marker='d', label='Aitken', color='tab:red')

axs[1].set_yscale('log')
axs[1].axhline(tolerancia_global, color='black', linestyle='--', linewidth=1.5, label=f'Tolerancia ($10^{{-6}}$)')
axs[1].set_title('Comparación de Convergencia (Error vs Iteración)')
axs[1].set_xlabel('Iteración (inicia en 0)')
axs[1].set_ylabel('Error Obtenido (escala logarítmica)')
axs[1].grid(True, which="both", ls="--", linewidth=0.5)
axs[1].legend()

plt.tight_layout()
plt.show()
