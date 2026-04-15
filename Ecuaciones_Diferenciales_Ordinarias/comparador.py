import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def resolver_edo_exacta(texto_dy_dx, x0_val, y0_val):
    """
    Resuelve la EDO analíticamente aplicando el problema de Cauchy.
    Soporta: sin, cos, tan, exp, sqrt, pi, e, etc.
    """
    x = sp.Symbol('x')
    y_func = sp.Function('y')(x)
    
    # Traducciones automáticas para español
    texto_dy_dx = texto_dy_dx.replace('sen', 'sin')
    texto_dy_dx = texto_dy_dx.replace('cos', 'cos')
    texto_dy_dx = texto_dy_dx.replace('tan', 'tan')
    texto_dy_dx = texto_dy_dx.replace('e^', 'exp(').replace('^', '**')
    
    # Diccionario con funciones matemáticas
    diccionario_local = {
        'y': y_func, 
        'e': sp.E, 
        'pi': sp.pi,
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'exp': sp.exp,
        'sqrt': sp.sqrt,
        'log': sp.log,
        'ln': sp.ln,
    }
    
    f_expr = sp.sympify(texto_dy_dx, locals=diccionario_local)
    
    edo = sp.Eq(y_func.diff(x), f_expr)
    condiciones_iniciales = {y_func.subs(x, x0_val): y0_val}
    
    solucion_general = sp.dsolve(edo, ics=condiciones_iniciales)
    expr_exacta = solucion_general.rhs
    
    f_exacta = sp.lambdify(x, expr_exacta, 'numpy')
    
    def f_exacta_segura(x_val):
        return float(f_exacta(x_val))
        
    return f_exacta_segura, expr_exacta

def euler(f, y0, x0, xf, h):
    """Método de Euler"""
    x_values = np.arange(x0, xf + h/2, h)
    x_values = np.round(x_values, decimals=10)
    n_steps = len(x_values)
    
    y_n = np.zeros(n_steps)
    y_n1 = np.zeros(n_steps)
    
    y_n[0] = y0
    
    for i in range(n_steps):
        if i > 0:
            y_n[i] = y_n1[i-1]
            
        y_n1[i] = y_n[i] + h * f(x_values[i], y_n[i])
        
    return x_values, y_n

def euler_mejorado(f, y0, x0, xf, h):
    """Método de Euler Mejorado (Heun)"""
    x_values = np.arange(x0, xf + h/2, h)
    x_values = np.round(x_values, decimals=10)
    n_steps = len(x_values)
    
    y_n = np.zeros(n_steps)
    y_corr = np.zeros(n_steps)
    
    y_n[0] = y0
    
    for i in range(n_steps):
        if i > 0:
            y_n[i] = y_corr[i-1]
            
        if i < n_steps - 1:
            x_act = x_values[i]
            y_act = y_n[i]
            x_sig = x_values[i+1]
            
            f_xy = f(x_act, y_act)
            y_pred = y_act + h * f_xy
            f_x_sig_y_pred = f(x_sig, y_pred)
            
            y_corr[i] = y_act + (h / 2.0) * (f_xy + f_x_sig_y_pred)
            
    return x_values, y_n

def runge_kutta_4(f, y0, x0, xf, h):
    """Método de Runge-Kutta de 4to Orden"""
    x_values = np.arange(x0, xf + h/2, h)
    x_values = np.round(x_values, decimals=10)
    n_steps = len(x_values)
    
    y_n = np.zeros(n_steps)
    y_n1 = np.zeros(n_steps)
    
    y_n[0] = y0
    
    for i in range(n_steps):
        if i > 0:
            y_n[i] = y_n1[i-1]
            
        if i < n_steps - 1:
            xn = x_values[i]
            yn = y_n[i]
            
            k1 = f(xn, yn)
            k2 = f(xn + h/2.0, yn + (h/2.0) * k1)
            k3 = f(xn + h/2.0, yn + (h/2.0) * k2)
            k4 = f(xn + h, yn + h * k3)
            
            y_n1[i] = yn + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
    return x_values, y_n

# ==========================================
# 1. PARÁMETROS DEL PROBLEMA
# ==========================================
ecuacion_str = "y*sin(x)"
y0 = 2
x0 = 0.0    # x inicial
xf = np.pi    # x final ← Hasta dónde calcular
h = np.pi/10    # paso (incremento)

decimales = 6

# Compilamos la función numérica
x_sym, y_sym = sp.symbols('x y')
ecuacion_str_limpia = ecuacion_str.replace('sen', 'sin').replace('e^', 'exp(').replace('^', '**')

diccionario_funciones = {
    'e': sp.E, 
    'pi': sp.pi,
    'sin': sp.sin,
    'cos': sp.cos,
    'tan': sp.tan,
    'exp': sp.exp,
    'sqrt': sp.sqrt,
    'log': sp.log,
    'ln': sp.ln,
}

f_num = sp.lambdify((x_sym, y_sym), sp.sympify(ecuacion_str_limpia, locals=diccionario_funciones), 'numpy')

# ==========================================
# 2. RESOLUCIÓN ANALÍTICA
# ==========================================
solucion_exacta, expr_exacta = resolver_edo_exacta(ecuacion_str, x0, y0)

# ==========================================
# 3. EJECUCIÓN DE LOS 3 MÉTODOS
# ==========================================
x_n, y_euler = euler(f_num, y0, x0, xf, h)
x_n, y_heun = euler_mejorado(f_num, y0, x0, xf, h)
x_n, y_rk4 = runge_kutta_4(f_num, y0, x0, xf, h)

# Solución exacta en cada punto
yr_values = np.array([solucion_exacta(xi) for xi in x_n])

# Errores
error_euler = np.abs(yr_values - y_euler)
error_heun = np.abs(yr_values - y_heun)
error_rk4 = np.abs(yr_values - y_rk4)

# ==========================================
# 4. TABLA COMPARATIVA
# ==========================================
w_n = 4
w_x = max(8, decimales + 2)
w_exact = max(12, decimales + 4)
w_euler_col = max(10, decimales + 3)
w_heun_col = max(10, decimales + 3)
w_rk4_col = max(10, decimales + 3)

separador_lines = w_n + w_x + w_exact + w_euler_col + w_heun_col + w_rk4_col + (6 * 3)

print("\n" + "="*separador_lines)
print(f"COMPARACIÓN DE MÉTODOS PARA ECUACIONES DIFERENCIALES ORDINARIAS")
print(f"EDO: dy/dx = {ecuacion_str} | y({x0}) = {y0} | h = {h}")
print(f"Solución Exacta (Y_r): y(x) = {expr_exacta}")
print("="*separador_lines)

print(f"{'n':<{w_n}} | {'x_n':<{w_x}} | {'Sol Exacta':<{w_exact}} | {'Euler':<{w_euler_col}} | {'Euler M':<{w_heun_col}} | {'Runge-K4':<{w_rk4_col}}")
print("-" * separador_lines)

for i in range(len(x_n)):
    xn_str = f"{x_n[i]:.{decimales}f}"
    yr_str = f"{yr_values[i]:.{decimales}f}"
    euler_str = f"{y_euler[i]:.{decimales}f}"
    heun_str = f"{y_heun[i]:.{decimales}f}"
    rk4_str = f"{y_rk4[i]:.{decimales}f}"
    
    print(f"{i:<{w_n}} | {xn_str:<{w_x}} | {yr_str:<{w_exact}} | {euler_str:<{w_euler_col}} | {heun_str:<{w_heun_col}} | {rk4_str:<{w_rk4_col}}")

print("="*separador_lines + "\n")

# ==========================================
# 5. TABLA DE ERRORES ABSOLUTOS
# ==========================================
print("="*separador_lines)
print(f"ERRORES ABSOLUTOS (|Y_r - y_n|)")
print("="*separador_lines)

w_err = max(14, decimales + 3)
separador_errors = w_n + w_x + (w_err * 3) + (4 * 3)

print(f"{'n':<{w_n}} | {'x_n':<{w_x}} | {'Error Euler':<{w_err}} | {'Error Heun':<{w_err}} | {'Error RK4':<{w_err}}")
print("-" * separador_errors)

for i in range(len(x_n)):
    xn_str = f"{x_n[i]:.{decimales}f}"
    
    if i == 0:
        err_euler_str = "-"
        err_heun_str = "-"
        err_rk4_str = "-"
    else:
        err_euler_str = f"{error_euler[i]:.2e}" if error_euler[i] < 1e-5 else f"{error_euler[i]:.{decimales}f}"
        err_heun_str = f"{error_heun[i]:.2e}" if error_heun[i] < 1e-5 else f"{error_heun[i]:.{decimales}f}"
        err_rk4_str = f"{error_rk4[i]:.2e}" if error_rk4[i] < 1e-5 else f"{error_rk4[i]:.{decimales}f}"
    
    print(f"{i:<{w_n}} | {xn_str:<{w_x}} | {err_euler_str:<{w_err}} | {err_heun_str:<{w_err}} | {err_rk4_str:<{w_err}}")

print("="*separador_errors + "\n")

# ==========================================
# 6. GRÁFICO COMPARATIVO
# ==========================================
plt.figure(figsize=(14, 8))

x_suave = np.linspace(x0, xf, 200)
y_suave = np.array([solucion_exacta(xi) for xi in x_suave])

# Línea de solución exacta (más gruesa)
plt.plot(x_suave, y_suave, 'k-', label='Solución Exacta', linewidth=3, zorder=5)

# Líneas de los métodos
plt.plot(x_n, y_euler, 'o--', label='Euler', color='#FF6B6B', markersize=6, linewidth=2, alpha=0.8)
plt.plot(x_n, y_heun, 's--', label='Euler Mejorado', color='#4ECDC4', markersize=6, linewidth=2, alpha=0.8)
plt.plot(x_n, y_rk4, '^--', label='Runge-Kutta 4', color='#45B7D1', markersize=6, linewidth=2, alpha=0.8)

# Líneas verticales de error (opcionales, como en la imagen)
for i in range(1, len(x_n)):
    plt.plot([x_n[i], x_n[i]], [y_euler[i], yr_values[i]], ':', color='#FF6B6B', alpha=0.3)

plt.title(f'Comparación de Métodos: dy/dx = {ecuacion_str} (h={h})', fontsize=14, fontweight='bold')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(loc='upper left', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==========================================
# 7. GRÁFICO DE ERRORES
# ==========================================
plt.figure(figsize=(12, 6))

plt.semilogy(x_n[1:], error_euler[1:], 'o-', label='Error Euler', color='#FF6B6B', linewidth=2, markersize=6)
plt.semilogy(x_n[1:], error_heun[1:], 's-', label='Error Euler Mejorado', color='#4ECDC4', linewidth=2, markersize=6)
plt.semilogy(x_n[1:], error_rk4[1:], '^-', label='Error Runge-Kutta 4', color='#45B7D1', linewidth=2, markersize=6)

plt.title(f'Evolución del Error Absoluto (escala logarítmica)', fontsize=14, fontweight='bold')
plt.xlabel('x', fontsize=12)
plt.ylabel('|Error|', fontsize=12)
plt.legend(loc='upper left', fontsize=11)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
