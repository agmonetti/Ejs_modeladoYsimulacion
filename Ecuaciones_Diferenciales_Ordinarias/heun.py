import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def resolver_edo_exacta(texto_dy_dx, x0_val, y0_val):
    x = sp.Symbol('x')
    y_func = sp.Function('y')(x)
    
    diccionario_local = {'y': y_func, 'e': sp.E, 'pi': sp.pi}
    texto_dy_dx = texto_dy_dx.replace('e^', 'exp(').replace('^', '**')
    f_expr = sp.sympify(texto_dy_dx, locals=diccionario_local)
    
    edo = sp.Eq(y_func.diff(x), f_expr)
    condiciones_iniciales = {y_func.subs(x, x0_val): y0_val}
    
    solucion_general = sp.dsolve(edo, ics=condiciones_iniciales)
    expr_exacta = solucion_general.rhs
    
    f_exacta = sp.lambdify(x, expr_exacta, 'numpy')
    
    def f_exacta_segura(x_val):
        return float(f_exacta(x_val))
        
    return f_exacta_segura, expr_exacta

def euler_mejorado(f, y0, x0, xf, h):
    x_values = np.arange(x0, xf + h/2, h)
    x_values = np.round(x_values, decimals=10)
    n_steps = len(x_values)
    
    y_n = np.zeros(n_steps)
    y_pred = np.zeros(n_steps)
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
            
            # Predictor
            y_pred[i] = y_act + h * f_xy
            f_x_sig_y_pred = f(x_sig, y_pred[i])
            
            # Corrector
            y_corr[i] = y_act + (h / 2.0) * (f_xy + f_x_sig_y_pred)
            
    return x_values, y_n, y_pred, y_corr

# ==========================================
# 1. PARÁMETROS DEL PROBLEMA
# ==========================================
ecuacion_str = "x + y"
x0 = 0.0
y0 = 1.0
xf = 1.0
h = 0.1
decimales = 4

x_sym, y_sym = sp.symbols('x y')
ecuacion_str_limpia = ecuacion_str.replace('e^', 'exp(').replace('^', '**')
f_num = sp.lambdify((x_sym, y_sym), sp.sympify(ecuacion_str_limpia, locals={'e': sp.E, 'pi': sp.pi}), 'numpy')

# ==========================================
# 2. RESOLUCIÓN ANALÍTICA
# ==========================================
solucion_exacta, expr_exacta = resolver_edo_exacta(ecuacion_str, x0, y0)

# ==========================================
# 3. EJECUCIÓN
# ==========================================
x_n, y_n, y_pred, y_corr = euler_mejorado(f_num, y0, x0, xf, h)

yr_values = np.array([solucion_exacta(xi) for xi in x_n])
error_values = np.abs(yr_values - y_n)

# ==========================================
# 4. IMPRESIÓN DE TABLA SÚPER BLINDADA
# ==========================================
# Anchos de columna dinámicos adaptados al texto y decimales
w_n = 4
w_x = max(6, decimales + 2)
w_y = max(10, decimales + 4)
w_pred = max(16, decimales + 4) # 'y*_{n+1} (Pred)'
w_corr = max(15, decimales + 4) # 'y_{n+1} (Corr)'
w_yr = max(10, decimales + 4)
w_err = max(11, decimales + 4)

# Sumamos todos los anchos más los espacios y los separadores " | "
separador_lines = w_n + w_x + w_y + w_pred + w_corr + w_yr + w_err + (6 * 3)

print("\n" + "="*separador_lines)
print(f"MÉTODO DE EULER MEJORADO (Heun)")
print(f"EDO: dy/dx = {ecuacion_str} | y({x0}) = {y0} | h = {h}")
print(f"Solución Exacta (Y_r): y(x) = {expr_exacta}")
print("="*separador_lines)

print(f"{'n':<{w_n}} | {'x_n':<{w_x}} | {'y_n':<{w_y}} | {'y*_{n+1} (Pred)':<{w_pred}} | {'y_{n+1} (Corr)':<{w_corr}} | {'Y_r':<{w_yr}} | {'ε (Error)':<{w_err}}")
print("-" * separador_lines)

for i in range(len(x_n)):
    xn_str = f"{x_n[i]:.{decimales}f}"
    
    yn_str = f"{y_n[i]:.{decimales}f}".rstrip('0').rstrip('.') if i==0 else f"{y_n[i]:.{decimales}f}"
    yr_str = f"{yr_values[i]:.{decimales}f}".rstrip('0').rstrip('.') if i==0 else f"{yr_values[i]:.{decimales}f}"
    
    if i < len(x_n) - 1:
        ypred_str = f"{y_pred[i]:.{decimales}f}"
        ycorr_str = f"{y_corr[i]:.{decimales}f}"
    else:
        ypred_str = "-"
        ycorr_str = "-"
    
    if i == 0:
        err_str = "-"
    else:
        err_str = f"{error_values[i]:.{decimales}f}"
        
    print(f"{i:<{w_n}} | {xn_str:<{w_x}} | {yn_str:<{w_y}} | {ypred_str:<{w_pred}} | {ycorr_str:<{w_corr}} | {yr_str:<{w_yr}} | {err_str:<{w_err}}")
print("="*separador_lines + "\n")

# ==========================================
# 5. GRÁFICO
# ==========================================
plt.figure(figsize=(10, 6))

x_suave = np.linspace(x0, xf, 200)
y_suave = np.array([solucion_exacta(xi) for xi in x_suave])

plt.plot(x_suave, y_suave, 'g-', label=f'Solución Exacta $Y_r$', linewidth=2)
plt.plot(x_n, y_n, 'bo-', label='Euler Mejorado $y_n$', markersize=5)

for i in range(1, len(x_n)):
    plt.plot([x_n[i], x_n[i]], [y_n[i], yr_values[i]], 'r:', alpha=0.6)

plt.title(f'Euler Mejorado vs Solución Exacta (h={h})', fontsize=14, fontweight='bold')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()