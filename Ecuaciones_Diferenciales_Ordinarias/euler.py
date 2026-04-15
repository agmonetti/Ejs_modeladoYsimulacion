import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def resolver_edo_exacta(texto_dy_dx, x0_val, y0_val):
    """
    Usa SymPy para resolver la EDO analíticamente aplicando el problema de Cauchy.
    """
    x = sp.Symbol('x')
    y_func = sp.Function('y')(x)
    
    # El truco clave: Le decimos a SymPy que la letra 'y' en tu texto es y(x)
    diccionario_local = {'y': y_func, 'e': sp.E, 'pi': sp.pi}
    texto_dy_dx = texto_dy_dx.replace('e^', 'exp(').replace('^', '**')
    f_expr = sp.sympify(texto_dy_dx, locals=diccionario_local)
    
    # Armamos la ecuación: y' = x + y(x)
    edo = sp.Eq(y_func.diff(x), f_expr)
    
    # Condición inicial: y(0) = 1
    condiciones_iniciales = {y_func.subs(x, x0_val): y0_val}
    
    # ¡Magia! Resuelve la EDO exacta
    solucion_general = sp.dsolve(edo, ics=condiciones_iniciales)
    expr_exacta = solucion_general.rhs
    
    f_exacta = sp.lambdify(x, expr_exacta, 'numpy')
    
    # Wrapper seguro para evitar que SymPy devuelva objetos raros
    def f_exacta_segura(x_val):
        res = f_exacta(x_val)
        return float(res)
        
    return f_exacta_segura, expr_exacta

def euler(f, y0, x0, xf, h):
    """
    Implementación del Método de Euler adaptada a la tabla del profesor.
    Devuelve los arreglos separados para x_n, y_n y y_{n+1}
    """
    # Genera x_values y redondea para evitar errores de punto flotante
    x_values = np.arange(x0, xf + h/2, h)
    x_values = np.round(x_values, decimals=10)  # Redondea a 10 decimales para limpiar errores
    n_steps = len(x_values)
    
    y_n = np.zeros(n_steps)
    y_n1 = np.zeros(n_steps) # Almacena la predicción y_{n+1}
    
    y_n[0] = y0
    
    for i in range(n_steps):
        if i > 0:
            y_n[i] = y_n1[i-1] # El y_{n+1} del paso anterior es nuestro nuevo y_n
            
        # Fórmula de Euler: y_{n+1} = y_n + h * f(x_n, y_n)
        y_n1[i] = y_n[i] + h * f(x_values[i], y_n[i])
        
    return x_values, y_n, y_n1

# ==========================================
# 1. PARÁMETROS DEL PROBLEMA
# ==========================================
ecuacion_str = "x + y"
x0 = 0.0
y0 = 1.0
xf = 1.0
h = 0.1
decimales = 8 # <--- Control de decimales a gusto

# Compilamos la función numérica
x_sym, y_sym = sp.symbols('x y')
ecuacion_str_limpia = ecuacion_str.replace('e^', 'exp(').replace('^', '**')
f_num = sp.lambdify((x_sym, y_sym), sp.sympify(ecuacion_str_limpia, locals={'e': sp.E, 'pi': sp.pi}), 'numpy')

# ==========================================
# 2. RESOLUCIÓN ANALÍTICA (Factor mu automático)
# ==========================================
solucion_exacta, expr_exacta = resolver_edo_exacta(ecuacion_str, x0, y0)

# ==========================================
# 3. EJECUCIÓN DEL MÉTODO DE EULER
# ==========================================
x_n, y_n, y_n1 = euler(f_num, y0, x0, xf, h)

# Calculamos los valores reales y el error iteración a iteración
yr_values = np.array([solucion_exacta(xi) for xi in x_n])
error_values = np.abs(yr_values - y_n)

# ==========================================
# 4. IMPRESIÓN DE TABLA (CLON DEL PIZARRÓN)
# ==========================================
# Calcula el ancho de columnas dinámicamente según decimales
col_width = max(10, decimales + 8)
separador_lines = 4 + col_width * 5 + 10  # Aproximación del ancho total

print("\n" + "="*separador_lines)
print(f"MÉTODO DE EULER")
print(f"EDO: dy/dx = {ecuacion_str} | y({x0}) = {y0} | h = {h}")
print(f"Solución Exacta (Y_r): y(x) = {expr_exacta}")
print("="*separador_lines)

# Encabezados clonados
print(f"{'n':<4} | {'x_n':<{col_width}} | {'y_n':<{col_width}} | {'y_n+1':<{col_width}} | {'Y_r':<{col_width}} | {'ε (Error)':<{col_width}}")
print("-" * separador_lines)

for i in range(len(x_n)):
    # Formateo con decimales para todas las columnas
    xn_str = f"{x_n[i]:.{decimales}f}"
    
    # La primera fila (n=0) lleva formato especial igual que en la pizarra
    yn_str = f"{y_n[i]:.{decimales}f}".rstrip('0').rstrip('.') if i==0 else f"{y_n[i]:.{decimales}f}"
    yr_str = f"{yr_values[i]:.{decimales}f}".rstrip('0').rstrip('.') if i==0 else f"{yr_values[i]:.{decimales}f}"
    yn1_str = f"{y_n1[i]:.{decimales}f}"
    
    if i == 0:
        err_str = "-"
    else:
        err_str = f"{error_values[i]:.{decimales}f}"
        
    print(f"{i:<4} | {xn_str:<{col_width}} | {yn_str:<{col_width}} | {yn1_str:<{col_width}} | {yr_str:<{col_width}} | {err_str:<{col_width}}")
print("="*separador_lines + "\n")

# ==========================================
# 5. GRÁFICO (MATPLOTLIB)
# ==========================================
plt.figure(figsize=(10, 6))

x_suave = np.linspace(x0, xf, 200)
y_suave = np.array([solucion_exacta(xi) for xi in x_suave])

plt.plot(x_suave, y_suave, 'g-', label=f'Solución Exacta $Y_r$', linewidth=2)
plt.plot(x_n, y_n, 'bo--', label='Aproximación Euler $y_n$', markersize=5)

for i in range(1, len(x_n)):
    plt.plot([x_n[i], x_n[i]], [y_n[i], yr_values[i]], 'r:', alpha=0.6)

plt.title(f'Método de Euler vs Solución Exacta (h={h})', fontsize=14, fontweight='bold')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()