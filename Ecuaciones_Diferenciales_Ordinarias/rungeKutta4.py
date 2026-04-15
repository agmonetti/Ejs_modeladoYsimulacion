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

def runge_kutta_4(f, y0, x0, xf, h):
    x_values = np.arange(x0, xf + h/2, h)
    x_values = np.round(x_values, decimals=10)
    n_steps = len(x_values)
    
    y_n = np.zeros(n_steps)
    k1_arr = np.zeros(n_steps)
    k2_arr = np.zeros(n_steps)
    k3_arr = np.zeros(n_steps)
    k4_arr = np.zeros(n_steps)
    y_n1 = np.zeros(n_steps)
    
    y_n[0] = y0
    
    for i in range(n_steps):
        if i > 0:
            y_n[i] = y_n1[i-1]
            
        if i < n_steps - 1:
            xn = x_values[i]
            yn = y_n[i]
            
            # Los 4 pasos de RK4
            k1 = f(xn, yn)
            k2 = f(xn + h/2.0, yn + (h/2.0) * k1)
            k3 = f(xn + h/2.0, yn + (h/2.0) * k2)
            k4 = f(xn + h, yn + h * k3)
            
            # Guardamos los k para la tabla
            k1_arr[i] = k1
            k2_arr[i] = k2
            k3_arr[i] = k3
            k4_arr[i] = k4
            
            # Promedio ponderado para el siguiente paso
            y_n1[i] = yn + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
    return x_values, y_n, k1_arr, k2_arr, k3_arr, k4_arr, y_n1

# ==========================================
# 1. PARÁMETROS DEL PROBLEMA
# ==========================================
ecuacion_str = "x + y"
x0 = 0.0
y0 = 1.0
xf = 1.0
h = 0.1
decimales = 5 # Usamos 5 decimales porque en el pizarrón el profe llega a 1.11034

x_sym, y_sym = sp.symbols('x y')
ecuacion_str_limpia = ecuacion_str.replace('e^', 'exp(').replace('^', '**')
f_num = sp.lambdify((x_sym, y_sym), sp.sympify(ecuacion_str_limpia, locals={'e': sp.E, 'pi': sp.pi}), 'numpy')

# ==========================================
# 2. RESOLUCIÓN ANALÍTICA
# ==========================================
solucion_exacta, expr_exacta = resolver_edo_exacta(ecuacion_str, x0, y0)

# ==========================================
# 3. EJECUCIÓN DEL MÉTODO RK4
# ==========================================
x_n, y_n, k1, k2, k3, k4, y_n1 = runge_kutta_4(f_num, y0, x0, xf, h)

yr_values = np.array([solucion_exacta(xi) for xi in x_n])
error_values = np.abs(yr_values - y_n)

# ==========================================
# 4. IMPRESIÓN DE LA SÚPER TABLA
# ==========================================
w_n = 4
w_x = max(6, decimales + 2)
w_y = max(8, decimales + 4)
w_k = max(8, decimales + 3) # Ancho para las k
w_yr = max(10, decimales + 4)
w_err = max(11, decimales + 4)

# Calculamos el ancho total sumando columnas y separadores
separador_lines = w_n + w_x + w_y + (w_k * 4) + w_y + w_yr + w_err + (9 * 3)

print("\n" + "="*separador_lines)
print(f"MÉTODO DE RUNGE-KUTTA DE 4TO ORDEN (RK4)")
print(f"EDO: dy/dx = {ecuacion_str} | y({x0}) = {y0} | h = {h}")
print(f"Solución Exacta (Y_r): y(x) = {expr_exacta}")
print("="*separador_lines)

print(f"{'n':<{w_n}} | {'x_n':<{w_x}} | {'y_n':<{w_y}} | {'k_1':<{w_k}} | {'k_2':<{w_k}} | {'k_3':<{w_k}} | {'k_4':<{w_k}} | {'y_n+1':<{w_y}} | {'Y_r':<{w_yr}} | {'ε (Error)':<{w_err}}")
print("-" * separador_lines)

for i in range(len(x_n)):
    xn_str = f"{x_n[i]:.{decimales}f}"
    
    yn_str = f"{y_n[i]:.{decimales}f}".rstrip('0').rstrip('.') if i==0 else f"{y_n[i]:.{decimales}f}"
    yr_str = f"{yr_values[i]:.{decimales}f}".rstrip('0').rstrip('.') if i==0 else f"{yr_values[i]:.{decimales}f}"
    
    if i < len(x_n) - 1:
        k1_str = f"{k1[i]:.{decimales}f}".rstrip('0').rstrip('.') if i==0 else f"{k1[i]:.{decimales}f}"
        k2_str = f"{k2[i]:.{decimales}f}".rstrip('0').rstrip('.') if i==0 else f"{k2[i]:.{decimales}f}"
        k3_str = f"{k3[i]:.{decimales}f}".rstrip('0').rstrip('.') if i==0 else f"{k3[i]:.{decimales}f}"
        k4_str = f"{k4[i]:.{decimales}f}".rstrip('0').rstrip('.') if i==0 else f"{k4[i]:.{decimales}f}"
        yn1_str = f"{y_n1[i]:.{decimales}f}"
    else:
        k1_str = "-"
        k2_str = "-"
        k3_str = "-"
        k4_str = "-"
        yn1_str = "-"
    
    if i == 0:
        err_str = "-"
    else:
        # Usamos notación científica si el error es ridículamente pequeño
        if error_values[i] < 1e-5:
            err_str = f"{error_values[i]:.2e}"
        else:
            err_str = f"{error_values[i]:.{decimales}f}"
        
    print(f"{i:<{w_n}} | {xn_str:<{w_x}} | {yn_str:<{w_y}} | {k1_str:<{w_k}} | {k2_str:<{w_k}} | {k3_str:<{w_k}} | {k4_str:<{w_k}} | {yn1_str:<{w_y}} | {yr_str:<{w_yr}} | {err_str:<{w_err}}")
print("="*separador_lines + "\n")

# ==========================================
# 5. GRÁFICO
# ==========================================
plt.figure(figsize=(10, 6))

x_suave = np.linspace(x0, xf, 200)
y_suave = np.array([solucion_exacta(xi) for xi in x_suave])

plt.plot(x_suave, y_suave, 'g-', label=f'Solución Exacta $Y_r$', linewidth=4, alpha=0.5)
plt.plot(x_n, y_n, 'b--', label='RK4 $y_n$', markersize=5, linewidth=2)

plt.title(f'Runge-Kutta 4 vs Solución Exacta (h={h})', fontsize=14, fontweight='bold')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()